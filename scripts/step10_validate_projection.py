"""
Step 10: 验证 MASt3R K+Rt 传统投影路线

用 MASt3R 估计的相机参数 (K, R, t)，通过 homography 将图像像素
投影到地面平面 (z=0)，评估地面坐标映射精度。

这绕开了 point maps，直接利用 MASt3R 优秀的全局几何估计
（焦距 1.5%、位置 0.08m）做传统投影。

用法:
    python scripts/step10_validate_projection.py --dataset multiviewx --frame_id 0
    python scripts/step10_validate_projection.py --dataset multiviewx --all_frames
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
    'wildtrack': {
        'num_cam': 7,
        'image_hw': (1080, 1920),
        'map_height': 36,
        'map_width': 12,
        'map_expand': 40,
    },
}

DATASET_ORIG_RESOLUTION = {
    'wildtrack':  (1080, 1920),
    'multiviewx': (1080, 1920),
}


def positionID_to_world_coord(pos_id, map_width=25, map_expand=40):
    grid_x = pos_id % (map_width * map_expand)
    grid_y = pos_id // (map_width * map_expand)
    return np.array([grid_x / map_expand, grid_y / map_expand])


def load_annotations(dataset, frame_id):
    params = DATASET_PARAMS[dataset]
    ds_name = "MultiviewX" if dataset == "multiviewx" else "Wildtrack"
    ann_path = ROOT / "data" / ds_name / "annotations_positions" / f"{frame_id + 1:05d}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    with open(ann_path) as f:
        raw = json.load(f)

    annotations = []
    for person in raw:
        world_xy = positionID_to_world_coord(
            person['positionID'], params['map_width'], params['map_expand'],
        )
        views = {}
        for v in person['views']:
            cam_id = v['viewNum']
            if v['xmin'] < 0:
                continue
            views[cam_id] = {
                'xmin': v['xmin'], 'ymin': v['ymin'],
                'xmax': v['xmax'], 'ymax': v['ymax'],
            }
        annotations.append({
            'personID': person['personID'],
            'world_xy': world_xy,
            'views': views,
        })
    return annotations


def load_pred_cameras(dataset, frame_id):
    """Load predicted camera parameters and rescale intrinsics to original resolution."""
    npz_path = ROOT / "results" / "vggt_predictions" / f"vggt_{dataset}_frame{frame_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Prediction not found: {npz_path}")

    data = np.load(str(npz_path))
    extrinsics = data['extrinsics']  # (N, 3, 4)
    intrinsics = data['intrinsics']  # (N, 3, 3)
    resized_hw = data['resized_hw'] if 'resized_hw' in data else None

    # Rescale intrinsics from internal resolution to original
    orig_hw = DATASET_ORIG_RESOLUTION[dataset]
    if resized_hw is not None:
        scale_x = orig_hw[1] / resized_hw[1]
        scale_y = orig_hw[0] / resized_hw[0]
        for i in range(intrinsics.shape[0]):
            intrinsics[i, 0, 0] *= scale_x   # fx
            intrinsics[i, 1, 1] *= scale_y   # fy
            intrinsics[i, 0, 2] *= scale_x   # cx
            intrinsics[i, 1, 2] *= scale_y   # cy

    cameras = {}
    for i in range(extrinsics.shape[0]):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        cameras[i] = {
            'K': intrinsics[i].copy(),
            'R': R,
            't': t,
            'center': extrinsic_to_camera_center(R, t),
        }
    return cameras


def load_gt_cameras(dataset):
    npz_path = ROOT / "results" / "gt_calibrations" / dataset / "gt_cameras.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"GT not found: {npz_path}")
    data = np.load(str(npz_path))
    cameras = {}
    cam_ids = sorted(set(int(k.split('_')[0].replace('cam', '')) for k in data.files))
    for cam_id in cam_ids:
        cameras[cam_id] = {
            'K': data[f"cam{cam_id}_intrinsic"],
            'R': data[f"cam{cam_id}_R"],
            't': data[f"cam{cam_id}_t"],
            'center': data[f"cam{cam_id}_center"],
        }
    return cameras


def align_cameras_sim3(pred_cameras, gt_cameras):
    """Sim(3) align predicted cameras to GT, return aligned cameras."""
    cam_ids = sorted(gt_cameras.keys())

    gt_positions = np.array([gt_cameras[i]['center'] for i in cam_ids])
    pred_positions = np.array([pred_cameras[i]['center'] for i in cam_ids])

    aligned_positions, sim3_params = align_poses_sim3(pred_positions, gt_positions)

    R_align = sim3_params['R']
    s = sim3_params['s']
    t_align = sim3_params['t']

    aligned_cameras = {}
    for idx, cam_id in enumerate(cam_ids):
        pred = pred_cameras[cam_id]
        R_aligned = pred['R'] @ R_align.T
        center_aligned = aligned_positions[idx]
        t_aligned = -R_aligned @ center_aligned

        aligned_cameras[cam_id] = {
            'K': pred['K'],
            'R': R_aligned,
            't': t_aligned,
            'center': center_aligned,
        }

    return aligned_cameras, sim3_params


def compute_ground_homography(K, R, t):
    """
    Compute homography from image plane to ground plane (z=0).

    For ground plane z=0, world point is [X, Y, 0].
    Projection: s * [u,v,1]^T = K @ [r1 r2 t] @ [X,Y,1]^T
    where r1,r2 are first two columns of R.

    So H_img2ground = inv(K @ [r1, r2, t])
    maps [u,v,1] → [X,Y,1] (ground coordinates).
    """
    r1 = R[:, 0]
    r2 = R[:, 1]
    H_ground2img = K @ np.column_stack([r1, r2, t])
    H_img2ground = np.linalg.inv(H_ground2img)
    return H_img2ground


def project_pixel_to_ground(px, py, H_img2ground):
    """Project pixel (px, py) to ground plane using homography."""
    p = H_img2ground @ np.array([px, py, 1.0])
    if abs(p[2]) < 1e-10:
        return np.array([np.nan, np.nan])
    return p[:2] / p[2]


def evaluate_projection(dataset, frame_id, use_gt=False):
    """
    Evaluate ground projection accuracy.

    Args:
        use_gt: If True, use GT cameras (sanity check). If False, use MASt3R predictions.
    """
    params = DATASET_PARAMS[dataset]
    source = "GT" if use_gt else "MASt3R"

    print(f"\n{'=' * 60}")
    print(f"Frame {frame_id}: Homography Ground Projection ({source})")
    print(f"{'=' * 60}")

    annotations = load_annotations(dataset, frame_id)
    gt_cameras = load_gt_cameras(dataset)

    if use_gt:
        eval_cameras = gt_cameras
        sim3_params = {'s': 1.0, 'R': np.eye(3), 't': np.zeros(3)}
    else:
        pred_cameras = load_pred_cameras(dataset, frame_id)
        eval_cameras, sim3_params = align_cameras_sim3(pred_cameras, gt_cameras)

    print(f"  Source: {source}")
    print(f"  Cameras: {len(eval_cameras)}")
    print(f"  People: {len(annotations)}")
    if not use_gt:
        print(f"  Sim(3) scale: {sim3_params['s']:.4f}")

    # Compute homography for each camera
    homographies = {}
    for cam_id, cam in eval_cameras.items():
        homographies[cam_id] = compute_ground_homography(cam['K'], cam['R'], cam['t'])

    # Project foot pixels to ground and compute errors
    observations = []
    for ann in annotations:
        for cam_id, bbox in ann['views'].items():
            if cam_id not in homographies:
                continue
            # Foot position = bottom center of bbox
            px = (bbox['xmin'] + bbox['xmax']) / 2.0
            py = bbox['ymax']

            ground_xy = project_pixel_to_ground(px, py, homographies[cam_id])
            if np.any(np.isnan(ground_xy)):
                continue

            gt_xy = ann['world_xy']
            error = np.linalg.norm(ground_xy - gt_xy)

            observations.append({
                'person_id': ann['personID'],
                'cam_id': cam_id,
                'gt_xy': gt_xy,
                'pred_xy': ground_xy,
                'error': error,
                'foot_pixel': (px, py),
            })

    if not observations:
        print("  [ERROR] No valid observations")
        return None

    errors = np.array([o['error'] for o in observations])

    # Per-camera stats
    per_camera = {}
    for cam_id in sorted(eval_cameras.keys()):
        cam_mask = np.array([o['cam_id'] == cam_id for o in observations])
        if cam_mask.sum() == 0:
            continue
        cam_errors = errors[cam_mask]
        per_camera[cam_id] = {
            'count': int(cam_mask.sum()),
            'mean': float(cam_errors.mean()),
            'median': float(np.median(cam_errors)),
            'max': float(cam_errors.max()),
        }
        print(f"  Camera {cam_id}: n={cam_mask.sum()}, "
              f"error mean={cam_errors.mean():.3f}m, "
              f"median={np.median(cam_errors):.3f}m, "
              f"max={cam_errors.max():.3f}m")

    # Cross-view consistency
    person_ids = set(o['person_id'] for o in observations)
    consistency = []
    for pid in person_ids:
        p_obs = [o for o in observations if o['person_id'] == pid]
        if len(p_obs) < 2:
            continue
        pred_xys = np.array([o['pred_xy'] for o in p_obs])
        centroid = pred_xys.mean(axis=0)
        devs = np.linalg.norm(pred_xys - centroid, axis=1)
        consistency.append(float(devs.mean()))

    # Summary
    print(f"\n  --- Summary ---")
    print(f"  Observations: {len(observations)}")
    print(f"  XY error: mean={errors.mean():.3f}m, "
          f"median={np.median(errors):.3f}m, max={errors.max():.3f}m")
    if consistency:
        print(f"  Cross-view consistency: {np.mean(consistency):.3f}m")

    # Gate
    GATE_XY = 0.5
    passed = errors.mean() < GATE_XY
    print(f"\n  --- Gate (XY < {GATE_XY}m) ---")
    print(f"  {'PASS' if passed else 'FAIL'} (mean={errors.mean():.3f}m)")

    result = {
        'source': source,
        'frame_id': frame_id,
        'num_observations': len(observations),
        'error_xy': {
            'mean': float(errors.mean()),
            'median': float(np.median(errors)),
            'max': float(errors.max()),
            'std': float(errors.std()),
        },
        'cross_view_consistency': float(np.mean(consistency)) if consistency else None,
        'per_camera': {str(k): v for k, v in per_camera.items()},
        'sim3_scale': float(sim3_params['s']),
        'gate_passed': bool(passed),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate K+Rt homography ground projection")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["wildtrack", "multiviewx"])
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use GT cameras as sanity check (should give ~0 error)")
    args = parser.parse_args()

    output_dir = ROOT / "results" / "projection_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_frames:
        pred_dir = ROOT / "results" / "vggt_predictions"
        frame_ids = sorted([
            int(p.stem.split('frame')[1])
            for p in pred_dir.glob(f"vggt_{args.dataset}_frame*.npz")
        ])
    else:
        frame_ids = [args.frame_id]

    print(f"Dataset: {args.dataset}")
    print(f"Frames: {frame_ids}")
    print(f"Source: {'GT (sanity check)' if args.use_gt else 'MASt3R predictions'}")

    all_results = []

    # First run GT sanity check if requested
    if args.use_gt:
        for fid in frame_ids:
            result = evaluate_projection(args.dataset, fid, use_gt=True)
            if result:
                all_results.append(result)
    else:
        for fid in frame_ids:
            result = evaluate_projection(args.dataset, fid, use_gt=False)
            if result:
                all_results.append(result)

    # Save results
    for r in all_results:
        tag = "gt" if r['source'] == 'GT' else "mast3r"
        path = output_dir / f"projection_{tag}_{args.dataset}_frame{r['frame_id']}.json"
        with open(path, 'w') as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {path}")

    # Multi-frame summary
    if len(all_results) > 1:
        means = [r['error_xy']['mean'] for r in all_results]
        print(f"\n{'=' * 60}")
        print(f"Multi-frame: mean={np.mean(means):.3f}m, std={np.std(means):.3f}m")
        gates = [r['gate_passed'] for r in all_results]
        print(f"Gate passed: {sum(gates)}/{len(gates)}")


if __name__ == "__main__":
    main()
