"""
Phase 1B Step 8: 验证 VGGT depth maps + 外参 的 pixel→ground 映射精度

Fallback 实验：world_points 直接映射失败 (3.614m > 1m 阈值)，
尝试用 depth map + 相机内外参 反投影到地面。

三种模式:
  A) depth + GT intrinsics + GT extrinsics  (upper bound, 测试 depth 本身质量)
  B) depth + GT intrinsics + VGGT extrinsics (Sim3-aligned)  (测试去标定可行性)
  C) depth + VGGT intrinsics (rescaled) + VGGT extrinsics (Sim3-aligned)  (完全免标定)

用法:
    python scripts/step8_validate_depth.py --dataset multiviewx --frame_id 0
    python scripts/step8_validate_depth.py --dataset multiviewx --all_frames
    python scripts/step8_validate_depth.py --dataset multiviewx --all_frames --mode all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.coord_transform import align_poses_sim3, extrinsic_to_camera_center

DATASET_PARAMS = {
    'multiviewx': {
        'num_cam': 6,
        'image_hw': (1080, 1920),
        'map_height': 16,
        'map_width': 25,
        'map_expand': 40,
    },
}


def positionID_to_world_coord(pos_id, map_width=25, map_expand=40):
    grid_x = pos_id % (map_width * map_expand)
    grid_y = pos_id // (map_width * map_expand)
    return np.array([grid_x / map_expand, grid_y / map_expand])


def load_annotations(dataset, frame_id):
    ann_path = ROOT / "data" / "MultiviewX" / "annotations_positions" / f"{frame_id + 1:05d}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")
    with open(ann_path) as f:
        raw = json.load(f)
    annotations = []
    for person in raw:
        world_xy = positionID_to_world_coord(person['positionID'],
                                              DATASET_PARAMS[dataset]['map_width'],
                                              DATASET_PARAMS[dataset]['map_expand'])
        views = {}
        for v in person['views']:
            if v['xmin'] < 0:
                continue
            views[v['viewNum']] = {
                'xmin': v['xmin'], 'ymin': v['ymin'],
                'xmax': v['xmax'], 'ymax': v['ymax'],
            }
        annotations.append({
            'personID': person['personID'],
            'world_xy': world_xy,
            'views': views,
        })
    return annotations


def load_gt_cameras(dataset):
    """Load GT camera parameters. Returns list of (K, R, t, center) per camera."""
    gt_path = ROOT / "results" / "gt_calibrations" / dataset / "gt_cameras.npz"
    data = np.load(str(gt_path))
    num_cam = DATASET_PARAMS[dataset]['num_cam']
    cameras = []
    for i in range(num_cam):
        K = data[f'cam{i}_intrinsic']
        R = data[f'cam{i}_R']
        t = data[f'cam{i}_t']
        center = data[f'cam{i}_center']
        cameras.append({'K': K, 'R': R, 't': t, 'center': center})
    return cameras


def load_vggt_data(dataset, frame_id):
    npz_path = ROOT / "results" / "vggt_predictions" / f"vggt_{dataset}_frame{frame_id}.npz"
    data = np.load(str(npz_path))
    return {
        'extrinsics': data['extrinsics'],       # (N, 3, 4)
        'intrinsics': data['intrinsics'],        # (N, 3, 3)
        'depth': data['depth'],                  # (N, H, W, 1)
        'resized_hw': data['resized_hw'],        # (2,)
    }


def rescale_intrinsics(K_vggt, resized_hw, original_hw):
    """Rescale VGGT intrinsics from resized resolution to original."""
    scale_x = original_hw[1] / resized_hw[1]
    scale_y = original_hw[0] / resized_hw[0]
    K = K_vggt.copy()
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    return K


def align_vggt_extrinsics(vggt_extrinsics, gt_cameras):
    """Align VGGT extrinsics to GT coordinate system using Sim(3) on camera centers."""
    num_cam = len(gt_cameras)
    vggt_centers = np.array([
        extrinsic_to_camera_center(vggt_extrinsics[i, :3, :3], vggt_extrinsics[i, :3, 3])
        for i in range(num_cam)
    ])
    gt_centers = np.array([cam['center'] for cam in gt_cameras])

    aligned_centers, sim3_params = align_poses_sim3(vggt_centers, gt_centers)

    # Align each camera's extrinsic
    aligned_extrinsics = []
    for i in range(num_cam):
        R_vggt = vggt_extrinsics[i, :3, :3]
        # Aligned rotation: R_aligned = R_vggt @ R_sim3^T
        R_aligned = R_vggt @ sim3_params['R'].T
        # Aligned translation from aligned center
        t_aligned = -R_aligned @ aligned_centers[i]
        E = np.hstack([R_aligned, t_aligned.reshape(3, 1)])
        aligned_extrinsics.append(E)

    return aligned_extrinsics, sim3_params


def get_depth_at_pixel(depth_map, px, py, image_hw, resized_hw, patch_size=2):
    """Get depth value at pixel (px, py) in original image coords."""
    scale_x = resized_hw[1] / image_hw[1]
    scale_y = resized_hw[0] / image_hw[0]
    vx = int(round(px * scale_x))
    vy = int(round(py * scale_y))
    H, W = int(resized_hw[0]), int(resized_hw[1])
    vx = np.clip(vx, patch_size, W - 1 - patch_size)
    vy = np.clip(vy, patch_size, H - 1 - patch_size)
    patch = depth_map[vy - patch_size:vy + patch_size + 1,
                      vx - patch_size:vx + patch_size + 1]
    return float(np.median(patch))


def pixel_depth_to_world(px, py, depth, K, R, t):
    """Unproject pixel + depth to world coordinate.

    pixel (px, py) + depth d → world 3D point
    x_cam = d * K^{-1} @ [px, py, 1]^T
    x_world = R^T @ (x_cam - t)
    """
    K_inv = np.linalg.inv(K)
    pixel_h = np.array([px, py, 1.0])
    x_cam = depth * K_inv @ pixel_h
    x_world = R.T @ (x_cam - t)
    return x_world


def evaluate_depth(dataset, frame_id, mode='A'):
    """
    Evaluate depth map accuracy for pixel→ground mapping.

    Modes:
      A: depth + GT K + GT [R|t]
      B: depth + GT K + VGGT [R|t] (Sim3-aligned)
      C: depth + VGGT K (rescaled) + VGGT [R|t] (Sim3-aligned)
    """
    params = DATASET_PARAMS[dataset]
    image_hw = params['image_hw']

    mode_names = {'A': 'GT_K + GT_Rt', 'B': 'GT_K + VGGT_Rt', 'C': 'VGGT_K + VGGT_Rt'}
    print(f"\n{'=' * 60}")
    print(f"Frame {frame_id} | Mode {mode}: {mode_names[mode]}")
    print(f"{'=' * 60}")

    annotations = load_annotations(dataset, frame_id)
    gt_cameras = load_gt_cameras(dataset)
    vggt = load_vggt_data(dataset, frame_id)
    depth_maps = vggt['depth'][:, :, :, 0]   # (N, H, W)
    resized_hw = tuple(vggt['resized_hw'].astype(int))

    # Prepare cameras based on mode
    if mode == 'A':
        cameras_K = [cam['K'] for cam in gt_cameras]
        cameras_R = [cam['R'] for cam in gt_cameras]
        cameras_t = [cam['t'] for cam in gt_cameras]
    elif mode == 'B':
        cameras_K = [cam['K'] for cam in gt_cameras]
        aligned_ext, sim3_p = align_vggt_extrinsics(vggt['extrinsics'], gt_cameras)
        cameras_R = [E[:3, :3] for E in aligned_ext]
        cameras_t = [E[:3, 3] for E in aligned_ext]
        print(f"  Sim(3) scale: {sim3_p['s']:.4f}")
    elif mode == 'C':
        cameras_K = [rescale_intrinsics(vggt['intrinsics'][i], resized_hw, image_hw)
                     for i in range(params['num_cam'])]
        aligned_ext, sim3_p = align_vggt_extrinsics(vggt['extrinsics'], gt_cameras)
        cameras_R = [E[:3, :3] for E in aligned_ext]
        cameras_t = [E[:3, 3] for E in aligned_ext]
        print(f"  Sim(3) scale: {sim3_p['s']:.4f}")

    # VGGT depth is NOT metric — values are in an arbitrary scale (~0.5 for
    # objects several meters away). We always project with raw depth then use
    # Sim(3) to align the resulting 3D point cloud to GT ground coordinates.
    # This tests whether depth is *proportionally* correct (good structure)
    # even if absolute scale is wrong.

    print(f"  Annotations: {len(annotations)} people")

    # Evaluate each person-camera pair
    observations = []
    for p_idx, ann in enumerate(annotations):
        for cam_id, bbox in ann['views'].items():
            px = (bbox['xmin'] + bbox['xmax']) / 2.0
            py = bbox['ymax']  # foot
            px = np.clip(px, 0, image_hw[1] - 1)
            py = np.clip(py, 0, image_hw[0] - 1)

            depth_val = get_depth_at_pixel(depth_maps[cam_id], px, py, image_hw, resized_hw)

            if depth_val <= 0 or np.isnan(depth_val):
                continue

            scaled_depth = depth_val  # raw depth, Sim(3) will handle scale
            world_pt = pixel_depth_to_world(
                px, py, scaled_depth,
                cameras_K[cam_id], cameras_R[cam_id], cameras_t[cam_id]
            )

            observations.append({
                'person_idx': p_idx,
                'cam_id': cam_id,
                'gt_xy': ann['world_xy'],
                'pred_xyz': world_pt,
                'depth_raw': depth_val,
                'depth_scaled': scaled_depth,
            })

    print(f"  Valid observations: {len(observations)}")
    if len(observations) < 4:
        print("  [ERROR] Too few observations")
        return None

    pred_xyz = np.array([o['pred_xyz'] for o in observations])
    gt_xy = np.array([o['gt_xy'] for o in observations])

    # VGGT depth is in arbitrary scale, so always use Sim(3) to align
    # the projected 3D point cloud to GT ground coordinates.
    # This tests structural/proportional accuracy of depth, not absolute scale.
    gt_3d = np.column_stack([gt_xy, np.zeros(len(gt_xy))])
    aligned_3d, final_sim3 = align_poses_sim3(pred_xyz, gt_3d)
    pred_xy = aligned_3d[:, :2]
    errors_xy = np.linalg.norm(pred_xy - gt_xy, axis=1)
    sim3_info = {'scale': float(final_sim3['s']),
                 'R': final_sim3['R'].tolist(),
                 't': final_sim3['t'].tolist()}
    print(f"  Point cloud Sim(3) scale: {final_sim3['s']:.4f}")

    # Per-camera stats
    per_camera = {}
    for cam_id in range(params['num_cam']):
        mask = np.array([o['cam_id'] == cam_id for o in observations])
        if mask.sum() == 0:
            continue
        cam_err = errors_xy[mask]
        depths = np.array([o['depth_raw'] for o in observations])[mask]
        per_camera[cam_id] = {
            'count': int(mask.sum()),
            'error_xy_mean': float(cam_err.mean()),
            'error_xy_median': float(np.median(cam_err)),
            'error_xy_max': float(cam_err.max()),
            'error_xy_std': float(cam_err.std()),
            'depth_mean': float(depths.mean()),
            'depth_std': float(depths.std()),
        }
        print(f"  Camera {cam_id}: n={mask.sum()}, "
              f"error_xy={cam_err.mean():.3f}m (med={np.median(cam_err):.3f}m), "
              f"depth={depths.mean():.2f}±{depths.std():.2f}")

    # Cross-view consistency
    person_consistency = []
    for p_idx in range(len(annotations)):
        p_mask = np.array([o['person_idx'] == p_idx for o in observations])
        if p_mask.sum() < 2:
            continue
        p_xy = pred_xy[p_mask]
        centroid = p_xy.mean(axis=0)
        devs = np.linalg.norm(p_xy - centroid, axis=1)
        person_consistency.append({
            'person_idx': p_idx,
            'num_views': int(p_mask.sum()),
            'mean_deviation': float(devs.mean()),
        })

    consistency_devs = [p['mean_deviation'] for p in person_consistency]

    # Z-height analysis (should be near 0 for ground-standing people)
    z_vals = pred_xyz[:, 2]

    # Summary
    summary = {
        'frame_id': frame_id,
        'mode': mode,
        'mode_desc': mode_names[mode],
        'num_people': len(annotations),
        'num_observations': len(observations),
        'error_xy': {
            'mean': float(errors_xy.mean()),
            'median': float(np.median(errors_xy)),
            'max': float(errors_xy.max()),
            'std': float(errors_xy.std()),
        },
        'z_height': {
            'mean': float(z_vals.mean()),
            'std': float(z_vals.std()),
            'min': float(z_vals.min()),
            'max': float(z_vals.max()),
        },
        'cross_view_consistency': {
            'mean_deviation': float(np.mean(consistency_devs)) if consistency_devs else float('nan'),
            'num_people_multi_view': len(person_consistency),
        },
        'depth_stats': {
            'mean': float(np.array([o['depth_raw'] for o in observations]).mean()),
            'std': float(np.array([o['depth_raw'] for o in observations]).std()),
        },
        'point_sim3': sim3_info,
    }

    # Gate check
    PASS_XY = 1.0
    PASS_CONSISTENCY = 1.0
    passed_xy = errors_xy.mean() < PASS_XY
    passed_consistency = (np.mean(consistency_devs) < PASS_CONSISTENCY) if consistency_devs else False

    print(f"\n  --- Summary (Mode {mode}) ---")
    print(f"  XY error: mean={errors_xy.mean():.3f}m, median={np.median(errors_xy):.3f}m")
    print(f"  Z height: mean={z_vals.mean():.3f}m, std={z_vals.std():.3f}m")
    if consistency_devs:
        print(f"  Cross-view consistency: {np.mean(consistency_devs):.3f}m")
    print(f"  Gate XY < {PASS_XY}m: {'PASS' if passed_xy else 'FAIL'} ({errors_xy.mean():.3f}m)")
    if consistency_devs:
        print(f"  Gate consistency < {PASS_CONSISTENCY}m: {'PASS' if passed_consistency else 'FAIL'} ({np.mean(consistency_devs):.3f}m)")

    return {
        'summary': summary,
        'per_camera': {str(k): v for k, v in per_camera.items()},
        'per_person_consistency': person_consistency,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate VGGT depth maps for ground projection")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["multiviewx"])
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="A=GT_K+GT_Rt, B=GT_K+VGGT_Rt, C=VGGT_K+VGGT_Rt, all=run all three")
    args = parser.parse_args()

    modes = ['A', 'B', 'C'] if args.mode == 'all' else [args.mode]

    print(f"Dataset: {args.dataset}")
    print(f"Depth Map Validation (Phase 1B)")
    print(f"Modes: {modes}")
    print("=" * 60)

    output_dir = ROOT / "results" / "depth_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find frames
    if args.all_frames:
        pred_dir = ROOT / "results" / "vggt_predictions"
        frame_ids = sorted([
            int(p.stem.split('frame')[1])
            for p in pred_dir.glob(f"vggt_{args.dataset}_frame*.npz")
        ])
    else:
        frame_ids = [args.frame_id]

    print(f"Frames: {frame_ids}")

    # Run evaluations
    all_results = {m: [] for m in modes}
    for fid in frame_ids:
        for mode in modes:
            result = evaluate_depth(args.dataset, fid, mode=mode)
            if result is None:
                continue
            all_results[mode].append(result)

            # Save per-frame result
            json_path = output_dir / f"depth_mode{mode}_{args.dataset}_frame{fid}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # Multi-frame summary per mode
    if len(frame_ids) > 1:
        print(f"\n{'=' * 60}")
        print(f"Multi-Frame Summary")
        print(f"{'=' * 60}")

        comparison = {}
        for mode in modes:
            results = all_results[mode]
            if not results:
                continue

            xy_means = [r['summary']['error_xy']['mean'] for r in results]
            xy_medians = [r['summary']['error_xy']['median'] for r in results]
            z_means = [r['summary']['z_height']['mean'] for r in results]
            consistency = [r['summary']['cross_view_consistency']['mean_deviation']
                          for r in results
                          if not np.isnan(r['summary']['cross_view_consistency']['mean_deviation'])]

            mode_summary = {
                'mode': mode,
                'mode_desc': results[0]['summary']['mode_desc'],
                'num_frames': len(results),
                'error_xy_mean': {
                    'across_frames_mean': float(np.mean(xy_means)),
                    'across_frames_std': float(np.std(xy_means)),
                    'across_frames_min': float(np.min(xy_means)),
                    'across_frames_max': float(np.max(xy_means)),
                },
                'error_xy_median': {
                    'across_frames_mean': float(np.mean(xy_medians)),
                },
                'z_height_mean': float(np.mean(z_means)),
                'cross_view_consistency': {
                    'across_frames_mean': float(np.mean(consistency)) if consistency else float('nan'),
                },
                'gate_pass_xy': float(np.mean(xy_means)) < 1.0,
                'per_frame': [r['summary'] for r in results],
            }

            comparison[mode] = mode_summary

            status = "PASS" if mode_summary['gate_pass_xy'] else "FAIL"
            print(f"\n  Mode {mode} ({results[0]['summary']['mode_desc']}):")
            print(f"    XY error: {np.mean(xy_means):.3f}m ± {np.std(xy_means):.3f}m  [{status}]")
            print(f"    Z height: {np.mean(z_means):.3f}m")
            if consistency:
                print(f"    Cross-view: {np.mean(consistency):.3f}m")

        # Save comparison
        json_path = output_dir / f"depth_comparison_{args.dataset}.json"
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n  Comparison saved: {json_path}")

        # Compare with world_points baseline
        print(f"\n  --- vs world_points baseline ---")
        print(f"  world_points XY error: 3.614m (from Phase 1)")
        for mode in modes:
            if mode in comparison:
                err = comparison[mode]['error_xy_mean']['across_frames_mean']
                improvement = (3.614 - err) / 3.614 * 100
                print(f"  Mode {mode}: {err:.3f}m ({improvement:+.1f}% vs world_points)")


if __name__ == "__main__":
    main()
