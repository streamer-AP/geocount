"""
Step 9: 多帧 VGGT 推理实验

测试给 VGGT 喂入多个时间帧是否能提升相机参数估计精度。
VGGT 支持任意数量的视角输入，这里把不同时间帧视为额外视角。

实验设计:
  Exp 1: 时间帧对 (frame_i, frame_j) 的 6+6=12 视角输入
  Exp 2: 3-帧窗口 (frame_i, i+1, i+2) 的 18 视角输入
  Exp 3: 5-帧窗口的 30 视角输入 (可能接近显存极限)

关键假设: 更多视角 → 更好的 SfM 基线 → 更准确的相机参数

用法:
    python scripts/step9_multi_frame_vggt.py --dataset multiviewx --exp pair
    python scripts/step9_multi_frame_vggt.py --dataset multiviewx --exp triple
    python scripts/step9_multi_frame_vggt.py --dataset multiviewx --exp window5
    python scripts/step9_multi_frame_vggt.py --dataset multiviewx --exp all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_PARAMS = {
    'multiviewx': {
        'num_cam': 6,
        'image_hw': (1080, 1920),
    },
}


def get_image_paths(dataset, frame_id, data_dir=None):
    """Get image paths for a single frame, all cameras."""
    if data_dir is None:
        ds_name = "MultiviewX" if dataset == "multiviewx" else "Wildtrack"
        data_dir = ROOT / "data" / ds_name
    data_dir = Path(data_dir)
    img_dir = data_dir / "Image_subsets"
    cam_dirs = sorted(img_dir.glob("C*"))
    paths = []
    for cam_dir in cam_dirs:
        frame_files = sorted(cam_dir.glob("*.png")) or sorted(cam_dir.glob("*.jpg"))
        if frame_id < len(frame_files):
            paths.append(str(frame_files[frame_id]))
    return paths


def run_vggt_multi_frame(image_paths_list, frame_ids, num_cam):
    """Run VGGT on multiple frames concatenated as views.

    image_paths_list: list of lists, each inner list = paths for one frame
    frame_ids: which frames were used
    num_cam: cameras per frame

    Returns predictions dict with per-frame camera params extracted.
    """
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Flatten all image paths
    all_paths = []
    for paths in image_paths_list:
        all_paths.extend(paths)

    total_views = len(all_paths)
    num_frames = len(frame_ids)
    print(f"\n  Multi-frame VGGT: {num_frames} frames × {num_cam} cameras = {total_views} views")
    for i, fid in enumerate(frame_ids):
        print(f"    Frame {fid}: {image_paths_list[i][0]} ... ({len(image_paths_list[i])} images)")

    # Device
    if torch.cuda.is_available():
        device = "cuda"
        use_autocast = True
        autocast_dtype = torch.bfloat16
    else:
        device = "cpu"
        use_autocast = False
        autocast_dtype = torch.float32

    print(f"  Device: {device}")

    # Load model
    print("  Loading VGGT model...")
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device)
    print(f"  Model loaded ({time.time() - t0:.1f}s)")

    # Load images
    print("  Preprocessing images...")
    images = load_and_preprocess_images(all_paths)
    images = images.unsqueeze(0).to(device)
    print(f"  Input tensor: {images.shape}")

    # Inference
    print("  Running inference...")
    t0 = time.time()
    with torch.no_grad():
        if use_autocast:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    inference_time = time.time() - t0
    print(f"  Inference done ({inference_time:.2f}s)")

    # Extract camera params
    pose_enc = predictions.get("pose_enc", None)
    if pose_enc is not None:
        image_shape = images.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_shape)
        extrinsics = extrinsics.cpu().numpy()
        intrinsics = intrinsics.cpu().numpy()
    else:
        extrinsics = predictions["extrinsic"].cpu().numpy()
        intrinsics = predictions["intrinsic"].cpu().numpy()

    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]

    # Extract depth and world_points
    depth = predictions.get('depth', None)
    world_points = predictions.get('world_points', None)
    world_points_conf = predictions.get('world_points_conf', None)

    if depth is not None:
        depth = depth.cpu().numpy()
        if depth.ndim > 4:
            depth = depth[0]
    if world_points is not None:
        world_points = world_points.cpu().numpy()
        if world_points.ndim > 4:
            world_points = world_points[0]
    if world_points_conf is not None:
        world_points_conf = world_points_conf.cpu().numpy()
        if world_points_conf.ndim > 3:
            world_points_conf = world_points_conf[0]

    # Split results per frame
    result = {
        'frame_ids': frame_ids,
        'num_frames': num_frames,
        'num_cam': num_cam,
        'total_views': total_views,
        'inference_time': inference_time,
        'resized_hw': np.array(images.shape[-2:]),
        # Full arrays (all views)
        'extrinsics_all': extrinsics,      # (total_views, 3, 4)
        'intrinsics_all': intrinsics,      # (total_views, 3, 3)
    }

    if depth is not None:
        result['depth_all'] = depth
    if world_points is not None:
        result['world_points_all'] = world_points
    if world_points_conf is not None:
        result['world_points_conf_all'] = world_points_conf

    # Per-frame splits
    for fi, fid in enumerate(frame_ids):
        start = fi * num_cam
        end = start + num_cam
        result[f'extrinsics_frame{fid}'] = extrinsics[start:end]
        result[f'intrinsics_frame{fid}'] = intrinsics[start:end]
        if depth is not None:
            result[f'depth_frame{fid}'] = depth[start:end]
        if world_points is not None:
            result[f'world_points_frame{fid}'] = world_points[start:end]
        if world_points_conf is not None:
            result[f'world_points_conf_frame{fid}'] = world_points_conf[start:end]

    # Free GPU memory
    del model, predictions, images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def evaluate_against_gt(extrinsics, intrinsics, dataset, resized_hw):
    """Quick evaluation of camera params against GT."""
    from utils.coord_transform import (
        align_poses_sim3,
        extrinsic_to_camera_center,
        relative_rotation_error,
    )

    gt_path = ROOT / "results" / "gt_calibrations" / dataset / "gt_cameras.npz"
    gt = np.load(str(gt_path))
    num_cam = DATASET_PARAMS[dataset]['num_cam']
    image_hw = DATASET_PARAMS[dataset]['image_hw']

    # GT params
    gt_centers = np.array([gt[f'cam{i}_center'] for i in range(num_cam)])
    gt_Rs = [gt[f'cam{i}_R'] for i in range(num_cam)]
    gt_Ks = [gt[f'cam{i}_intrinsic'] for i in range(num_cam)]

    # VGGT params
    vggt_centers = np.array([
        extrinsic_to_camera_center(extrinsics[i, :3, :3], extrinsics[i, :3, 3])
        for i in range(num_cam)
    ])

    # Sim(3) alignment
    aligned_centers, sim3_p = align_poses_sim3(vggt_centers, gt_centers)
    position_errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)

    # Relative rotation errors
    rel_rot_errors = []
    for i in range(num_cam):
        for j in range(i + 1, num_cam):
            err = relative_rotation_error(
                extrinsics[i, :3, :3], extrinsics[j, :3, :3],
                gt_Rs[i], gt_Rs[j]
            )
            rel_rot_errors.append(err)

    # Intrinsic errors (rescale to original)
    focal_errors = []
    for i in range(num_cam):
        K_rescaled = intrinsics[i].copy()
        scale_x = image_hw[1] / resized_hw[1]
        scale_y = image_hw[0] / resized_hw[0]
        K_rescaled[0, 0] *= scale_x
        K_rescaled[1, 1] *= scale_y
        gt_fx = gt_Ks[i][0, 0]
        gt_fy = gt_Ks[i][1, 1]
        pred_fx = K_rescaled[0, 0]
        pred_fy = K_rescaled[1, 1]
        fx_err = abs(pred_fx - gt_fx) / gt_fx * 100
        fy_err = abs(pred_fy - gt_fy) / gt_fy * 100
        focal_errors.append((fx_err + fy_err) / 2)

    return {
        'position_error_mean': float(np.mean(position_errors)),
        'position_error_per_cam': position_errors.tolist(),
        'rel_rotation_error_mean': float(np.mean(rel_rot_errors)),
        'rel_rotation_error_per_pair': [float(e) for e in rel_rot_errors],
        'focal_error_mean': float(np.mean(focal_errors)),
        'focal_error_per_cam': [float(e) for e in focal_errors],
        'sim3_scale': float(sim3_p['s']),
    }


def save_results(result, output_dir, name):
    """Save experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save NPZ (arrays)
    npz_data = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            npz_data[k] = v
        elif isinstance(v, (list, int, float)):
            npz_data[k] = np.array(v)
    npz_path = output_dir / f"{name}.npz"
    np.savez(str(npz_path), **npz_data)
    print(f"  Saved: {npz_path}")
    return npz_path


def run_experiment(dataset, exp_type):
    """Run a multi-frame experiment."""
    num_cam = DATASET_PARAMS[dataset]['num_cam']
    output_dir = ROOT / "results" / "multi_frame_vggt"

    if exp_type == 'pair':
        # Test frame pairs: (0,5), (0,9), (2,7) — diverse temporal gaps
        pairs = [(0, 5), (0, 9), (2, 7), (3, 8)]
        all_evals = []

        for fi, fj in pairs:
            name = f"pair_{dataset}_f{fi}_f{fj}"
            print(f"\n{'=' * 60}")
            print(f"Experiment: Frame pair ({fi}, {fj})")
            print(f"{'=' * 60}")

            paths_i = get_image_paths(dataset, fi)
            paths_j = get_image_paths(dataset, fj)
            if not paths_i or not paths_j:
                print(f"  [SKIP] Missing images")
                continue

            result = run_vggt_multi_frame([paths_i, paths_j], [fi, fj], num_cam)
            save_results(result, output_dir, name)

            # Evaluate each frame's cameras from the multi-frame result
            for fid in [fi, fj]:
                ext = result[f'extrinsics_frame{fid}']
                intr = result[f'intrinsics_frame{fid}']
                resized_hw = result['resized_hw']
                ev = evaluate_against_gt(ext, intr, dataset, resized_hw)
                ev['frame_id'] = fid
                ev['source'] = f'pair({fi},{fj})'
                ev['inference_time'] = result['inference_time']
                all_evals.append(ev)

                print(f"\n  Frame {fid} from pair({fi},{fj}):")
                print(f"    Position error: {ev['position_error_mean']:.3f}m")
                print(f"    Rotation error: {ev['rel_rotation_error_mean']:.2f}°")
                print(f"    Focal error: {ev['focal_error_mean']:.1f}%")

        # Save summary
        json_path = output_dir / f"pair_summary_{dataset}.json"
        with open(json_path, 'w') as f:
            json.dump(all_evals, f, indent=2)
        print(f"\n  Pair summary: {json_path}")
        return all_evals

    elif exp_type == 'triple':
        # 3-frame windows
        triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 4, 9)]
        all_evals = []

        for frames in triples:
            name = f"triple_{dataset}_f{'_f'.join(map(str, frames))}"
            print(f"\n{'=' * 60}")
            print(f"Experiment: Frame triple {frames}")
            print(f"{'=' * 60}")

            paths_list = [get_image_paths(dataset, fid) for fid in frames]
            if any(not p for p in paths_list):
                print(f"  [SKIP] Missing images")
                continue

            result = run_vggt_multi_frame(paths_list, list(frames), num_cam)
            save_results(result, output_dir, name)

            for fid in frames:
                ext = result[f'extrinsics_frame{fid}']
                intr = result[f'intrinsics_frame{fid}']
                resized_hw = result['resized_hw']
                ev = evaluate_against_gt(ext, intr, dataset, resized_hw)
                ev['frame_id'] = fid
                ev['source'] = f'triple{frames}'
                ev['inference_time'] = result['inference_time']
                all_evals.append(ev)

                print(f"\n  Frame {fid} from triple{frames}:")
                print(f"    Position error: {ev['position_error_mean']:.3f}m")
                print(f"    Rotation error: {ev['rel_rotation_error_mean']:.2f}°")
                print(f"    Focal error: {ev['focal_error_mean']:.1f}%")

        json_path = output_dir / f"triple_summary_{dataset}.json"
        with open(json_path, 'w') as f:
            json.dump(all_evals, f, indent=2)
        print(f"\n  Triple summary: {json_path}")
        return all_evals

    elif exp_type == 'window5':
        # 5-frame windows (30 views — may be heavy on memory)
        windows = [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)]
        all_evals = []

        for frames in windows:
            name = f"window5_{dataset}_f{'_f'.join(map(str, frames))}"
            print(f"\n{'=' * 60}")
            print(f"Experiment: 5-frame window {frames}")
            print(f"{'=' * 60}")

            paths_list = [get_image_paths(dataset, fid) for fid in frames]
            if any(not p for p in paths_list):
                print(f"  [SKIP] Missing images")
                continue

            try:
                result = run_vggt_multi_frame(paths_list, list(frames), num_cam)
                save_results(result, output_dir, name)

                for fid in frames:
                    ext = result[f'extrinsics_frame{fid}']
                    intr = result[f'intrinsics_frame{fid}']
                    resized_hw = result['resized_hw']
                    ev = evaluate_against_gt(ext, intr, dataset, resized_hw)
                    ev['frame_id'] = fid
                    ev['source'] = f'window5{frames}'
                    ev['inference_time'] = result['inference_time']
                    all_evals.append(ev)

                    print(f"\n  Frame {fid} from window{frames}:")
                    print(f"    Position error: {ev['position_error_mean']:.3f}m")
                    print(f"    Rotation error: {ev['rel_rotation_error_mean']:.2f}°")
                    print(f"    Focal error: {ev['focal_error_mean']:.1f}%")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] 30 views too many, skipping")
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        if all_evals:
            json_path = output_dir / f"window5_summary_{dataset}.json"
            with open(json_path, 'w') as f:
                json.dump(all_evals, f, indent=2)
            print(f"\n  Window5 summary: {json_path}")
        return all_evals


def main():
    parser = argparse.ArgumentParser(description="Multi-frame VGGT inference experiments")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["multiviewx"])
    parser.add_argument("--exp", type=str, default="all",
                        choices=["pair", "triple", "window5", "all"],
                        help="Experiment type")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Multi-frame VGGT Experiments")
    print("=" * 60)

    experiments = ['pair', 'triple', 'window5'] if args.exp == 'all' else [args.exp]

    all_results = {}
    for exp in experiments:
        print(f"\n\n{'#' * 60}")
        print(f"# Experiment: {exp}")
        print(f"{'#' * 60}")
        evals = run_experiment(args.dataset, exp)
        if evals:
            all_results[exp] = evals

    # Final comparison with single-frame baseline
    print(f"\n\n{'#' * 60}")
    print("# COMPARISON: Single-frame vs Multi-frame")
    print(f"{'#' * 60}")

    # Load single-frame baseline
    baseline_path = ROOT / "results" / "evaluation" / f"multi_frame_summary_{args.dataset}.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\n  Single-frame baseline (10 frames):")
        print(f"    Position error: {baseline.get('position_error', {}).get('mean', 'N/A')}")
        print(f"    Rotation error: {baseline.get('relative_rotation_error', {}).get('mean', 'N/A')}")
        print(f"    Focal error: {baseline.get('focal_length_error', {}).get('mean', 'N/A')}")

    for exp, evals in all_results.items():
        rot_errors = [e['rel_rotation_error_mean'] for e in evals]
        pos_errors = [e['position_error_mean'] for e in evals]
        focal_errors = [e['focal_error_mean'] for e in evals]
        print(f"\n  {exp} ({len(evals)} evaluations):")
        print(f"    Position error: {np.mean(pos_errors):.3f}m ± {np.std(pos_errors):.3f}m")
        print(f"    Rotation error: {np.mean(rot_errors):.2f}° ± {np.std(rot_errors):.2f}°")
        print(f"    Focal error: {np.mean(focal_errors):.1f}% ± {np.std(focal_errors):.1f}%")

    # Save full comparison
    output_dir = ROOT / "results" / "multi_frame_vggt"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"full_comparison_{args.dataset}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full comparison: {json_path}")


if __name__ == "__main__":
    main()
