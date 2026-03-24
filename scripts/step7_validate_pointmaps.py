"""
Phase 1 Step 7: 验证 VGGT world_points 的 pixel→ground 映射精度

这是 PDE-VIC 路线的 GATE 验证：VGGT world_points 是否能提供可用的
pixel → 地面坐标映射（误差 < 1m）。

处理流程:
1. 从 GT 标注提取每人的 positionID → 世界坐标
2. 从 bbox 底部中点确定脚部像素，缩放到 VGGT 分辨率 (518×294)
3. 在 VGGT world_points 中查找对应像素的 3D 坐标
4. 用 Sim(3) 对齐将 VGGT 坐标系注册到 GT 世界坐标系
5. 计算对齐后 3D 点与 GT 地面位置的误差

用法:
    python scripts/step7_validate_pointmaps.py --dataset multiviewx --frame_id 0
    python scripts/step7_validate_pointmaps.py --dataset multiviewx --all_frames
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.coord_transform import align_poses_sim3


# Dataset parameters
DATASET_PARAMS = {
    'multiviewx': {
        'num_cam': 6,
        'image_hw': (1080, 1920),
        'map_height': 16,   # meters
        'map_width': 25,    # meters
        'map_expand': 40,   # grid cells per meter
    },
}


def positionID_to_world_coord(pos_id, map_width=25, map_expand=40):
    """Convert MultiviewX positionID to world coordinates (meters)."""
    grid_x = pos_id % (map_width * map_expand)
    grid_y = pos_id // (map_width * map_expand)
    coord_x = grid_x / map_expand
    coord_y = grid_y / map_expand
    return np.array([coord_x, coord_y])


def load_annotations(dataset, frame_id):
    """Load GT annotations for a frame.

    Returns list of dicts with 'world_xy' (2D ground coord) and 'views' (bbox per camera).
    """
    params = DATASET_PARAMS[dataset]
    # Annotation files are 1-indexed: frame_id 0 → 00001.json
    ann_path = ROOT / "data" / "MultiviewX" / "annotations_positions" / f"{frame_id + 1:05d}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    with open(ann_path) as f:
        raw = json.load(f)

    annotations = []
    for person in raw:
        world_xy = positionID_to_world_coord(
            person['positionID'],
            params['map_width'],
            params['map_expand'],
        )
        views = {}
        for v in person['views']:
            cam_id = v['viewNum']
            # Skip invisible (bbox = -1)
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


def load_vggt_pointmaps(dataset, frame_id):
    """Load world_points and confidence from NPZ (VGGT or MASt3R)."""
    npz_path = ROOT / "results" / "vggt_predictions" / f"vggt_{dataset}_frame{frame_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Prediction not found: {npz_path}\n"
            f"Run: python scripts/step4_run_vggt.py (or step4_run_dust3r.py) --dataset {dataset} --frame_id {frame_id}"
        )
    data = np.load(str(npz_path))
    result = {
        'world_points': data['world_points'],       # (N_cam, H, W, 3)
        'resized_hw': data['resized_hw'],            # (2,)
    }
    # world_points_conf is optional (MASt3R doesn't produce it)
    if 'world_points_conf' in data:
        result['world_points_conf'] = data['world_points_conf']
    else:
        # Use ones as placeholder confidence
        result['world_points_conf'] = np.ones(data['world_points'].shape[:3], dtype=np.float32)
    return result


def get_foot_pixel(bbox, image_hw=(1080, 1920)):
    """Get foot position (bottom center of bbox) in original image coordinates."""
    x_center = (bbox['xmin'] + bbox['xmax']) / 2.0
    y_foot = bbox['ymax']  # bottom of bbox = feet
    # Clamp to image bounds
    x_center = np.clip(x_center, 0, image_hw[1] - 1)
    y_foot = np.clip(y_foot, 0, image_hw[0] - 1)
    return x_center, y_foot


def pixel_to_vggt_coords(px, py, image_hw, resized_hw):
    """Scale pixel coordinates from original image to VGGT resolution."""
    scale_x = resized_hw[1] / image_hw[1]  # 518/1920
    scale_y = resized_hw[0] / image_hw[0]  # 294/1080
    vx = px * scale_x
    vy = py * scale_y
    return vx, vy


def lookup_world_point(world_points, vx, vy, patch_size=2):
    """Look up 3D point from world_points map at (vx, vy).

    Uses a small patch around the pixel and takes the median to reduce noise.
    world_points: (H, W, 3)
    """
    H, W, _ = world_points.shape
    # Round to nearest pixel
    ix = int(round(vx))
    iy = int(round(vy))
    # Clamp
    ix = np.clip(ix, patch_size, W - 1 - patch_size)
    iy = np.clip(iy, patch_size, H - 1 - patch_size)

    patch = world_points[iy - patch_size:iy + patch_size + 1,
                         ix - patch_size:ix + patch_size + 1, :]  # (2p+1, 2p+1, 3)
    return np.median(patch.reshape(-1, 3), axis=0)


def lookup_confidence(conf_map, vx, vy):
    """Look up confidence at (vx, vy)."""
    H, W = conf_map.shape
    ix = int(round(np.clip(vx, 0, W - 1)))
    iy = int(round(np.clip(vy, 0, H - 1)))
    return float(conf_map[iy, ix])


def align_points_sim3(vggt_points_3d, gt_points_2d):
    """Align VGGT 3D points to GT 2D ground coordinates using Sim(3).

    Since GT is 2D (ground plane), we add z=0 for GT and perform 3D Sim(3) alignment.
    Returns aligned 3D points and Sim(3) params.
    """
    # GT points: add z=0 (ground plane)
    gt_3d = np.column_stack([gt_points_2d, np.zeros(len(gt_points_2d))])

    # Use Sim(3) alignment: find s, R, t such that gt ≈ s * R @ vggt + t
    aligned, params = align_poses_sim3(vggt_points_3d, gt_3d)
    return aligned, params


def evaluate_pointmaps(dataset, frame_id, conf_threshold=None):
    """Main evaluation for a single frame.

    Returns dict with per-person, per-camera, and summary metrics.
    """
    params = DATASET_PARAMS[dataset]
    image_hw = params['image_hw']
    resized_hw_expected = (294, 518)

    print(f"\n{'=' * 60}")
    print(f"Frame {frame_id}: Point Maps Validation")
    print(f"{'=' * 60}")

    # Load data
    annotations = load_annotations(dataset, frame_id)
    vggt = load_vggt_pointmaps(dataset, frame_id)
    world_points = vggt['world_points']
    world_points_conf = vggt['world_points_conf']
    resized_hw = tuple(vggt['resized_hw'].astype(int))

    print(f"  Annotations: {len(annotations)} people")
    print(f"  VGGT resolution: {resized_hw}")

    # Step 1: Extract foot pixels and VGGT 3D points for each person-camera pair
    observations = []  # list of (person_idx, cam_id, vggt_3d, gt_xy, conf)
    for p_idx, ann in enumerate(annotations):
        for cam_id, bbox in ann['views'].items():
            px, py = get_foot_pixel(bbox, image_hw)
            vx, vy = pixel_to_vggt_coords(px, py, image_hw, resized_hw)

            pt_3d = lookup_world_point(world_points[cam_id], vx, vy)
            conf = lookup_confidence(world_points_conf[cam_id], vx, vy)

            if conf_threshold is not None and conf < conf_threshold:
                continue

            observations.append({
                'person_idx': p_idx,
                'cam_id': cam_id,
                'vggt_3d': pt_3d,
                'gt_xy': ann['world_xy'],
                'conf': conf,
                'foot_pixel': (px, py),
            })

    print(f"  Valid observations: {len(observations)}")

    if len(observations) < 4:
        print("  [ERROR] Too few observations for Sim(3) alignment")
        return None

    # Step 2: Sim(3) alignment using all observations
    vggt_points = np.array([o['vggt_3d'] for o in observations])
    gt_points_2d = np.array([o['gt_xy'] for o in observations])

    aligned_3d, sim3_params = align_points_sim3(vggt_points, gt_points_2d)
    print(f"  Sim(3) scale: {sim3_params['s']:.4f}")

    # Step 3: Compute errors
    gt_3d = np.column_stack([gt_points_2d, np.zeros(len(gt_points_2d))])
    errors_3d = np.linalg.norm(aligned_3d - gt_3d, axis=1)
    errors_xy = np.linalg.norm(aligned_3d[:, :2] - gt_3d[:, :2], axis=1)

    # Per-camera statistics
    per_camera = {}
    for cam_id in range(params['num_cam']):
        cam_mask = np.array([o['cam_id'] == cam_id for o in observations])
        if cam_mask.sum() == 0:
            continue
        cam_errors = errors_xy[cam_mask]
        cam_confs = np.array([o['conf'] for o in observations])[cam_mask]
        per_camera[cam_id] = {
            'count': int(cam_mask.sum()),
            'error_xy_mean': float(cam_errors.mean()),
            'error_xy_median': float(np.median(cam_errors)),
            'error_xy_max': float(cam_errors.max()),
            'error_xy_std': float(cam_errors.std()),
            'conf_mean': float(cam_confs.mean()),
        }
        print(f"  Camera {cam_id}: n={cam_mask.sum()}, "
              f"error_xy mean={cam_errors.mean():.3f}m, "
              f"median={np.median(cam_errors):.3f}m, "
              f"max={cam_errors.max():.3f}m")

    # Cross-view consistency: for each person, compute std of projected XY across views
    person_consistency = []
    for p_idx, ann in enumerate(annotations):
        p_mask = np.array([o['person_idx'] == p_idx for o in observations])
        if p_mask.sum() < 2:
            continue
        p_aligned = aligned_3d[p_mask, :2]
        # Mean deviation from centroid
        centroid = p_aligned.mean(axis=0)
        deviations = np.linalg.norm(p_aligned - centroid, axis=1)
        person_consistency.append({
            'person_idx': p_idx,
            'num_views': int(p_mask.sum()),
            'mean_deviation': float(deviations.mean()),
            'max_deviation': float(deviations.max()),
        })

    consistency_devs = [p['mean_deviation'] for p in person_consistency]

    # BEV grid coverage
    grid_dx = 0.5
    grid_nx = int(params['map_width'] / grid_dx)
    grid_ny = int(params['map_height'] / grid_dx)
    coverage_grid = np.zeros((grid_ny, grid_nx), dtype=bool)
    for pt in aligned_3d[:, :2]:
        gx = int(pt[0] / grid_dx)
        gy = int(pt[1] / grid_dx)
        if 0 <= gx < grid_nx and 0 <= gy < grid_ny:
            coverage_grid[gy, gx] = True
    coverage_pct = coverage_grid.sum() / coverage_grid.size * 100

    # Summary
    summary = {
        'frame_id': frame_id,
        'num_people': len(annotations),
        'num_observations': len(observations),
        'error_xy': {
            'mean': float(errors_xy.mean()),
            'median': float(np.median(errors_xy)),
            'max': float(errors_xy.max()),
            'std': float(errors_xy.std()),
        },
        'error_3d': {
            'mean': float(errors_3d.mean()),
            'median': float(np.median(errors_3d)),
            'max': float(errors_3d.max()),
            'std': float(errors_3d.std()),
        },
        'cross_view_consistency': {
            'mean_deviation': float(np.mean(consistency_devs)) if consistency_devs else float('nan'),
            'max_deviation': float(np.max(consistency_devs)) if consistency_devs else float('nan'),
            'num_people_multi_view': len(person_consistency),
        },
        'bev_coverage_pct': float(coverage_pct),
        'sim3_params': {
            'scale': float(sim3_params['s']),
            'R': sim3_params['R'].tolist(),
            't': sim3_params['t'].tolist(),
        },
    }

    print(f"\n  --- Summary ---")
    print(f"  Ground projection error (XY): mean={errors_xy.mean():.3f}m, "
          f"median={np.median(errors_xy):.3f}m, max={errors_xy.max():.3f}m")
    print(f"  3D error: mean={errors_3d.mean():.3f}m, "
          f"median={np.median(errors_3d):.3f}m")
    print(f"  Cross-view consistency: mean deviation={np.mean(consistency_devs):.3f}m" if consistency_devs else "  Cross-view consistency: N/A")
    print(f"  BEV grid coverage (0.5m cells): {coverage_pct:.1f}%")

    # Pass/fail assessment
    PASS_XY = 1.0
    PASS_COVERAGE = 80.0
    PASS_CONSISTENCY = 1.0

    passed_xy = errors_xy.mean() < PASS_XY
    passed_consistency = (np.mean(consistency_devs) < PASS_CONSISTENCY) if consistency_devs else False

    print(f"\n  --- Gate Criteria ---")
    print(f"  XY error < {PASS_XY}m: {'PASS' if passed_xy else 'FAIL'} ({errors_xy.mean():.3f}m)")
    print(f"  BEV coverage > {PASS_COVERAGE}%: {'PASS' if coverage_pct > PASS_COVERAGE else 'FAIL'} ({coverage_pct:.1f}%)")
    print(f"  Cross-view < {PASS_CONSISTENCY}m: {'PASS' if passed_consistency else 'FAIL'} "
          f"({np.mean(consistency_devs):.3f}m)" if consistency_devs else
          f"  Cross-view < {PASS_CONSISTENCY}m: N/A")

    results = {
        'summary': summary,
        'per_camera': {str(k): v for k, v in per_camera.items()},
        'per_person_consistency': person_consistency,
        'observations': [
            {
                'person_idx': o['person_idx'],
                'cam_id': o['cam_id'],
                'gt_xy': o['gt_xy'].tolist(),
                'vggt_3d': o['vggt_3d'].tolist(),
                'aligned_3d': aligned_3d[i].tolist(),
                'error_xy': float(errors_xy[i]),
                'conf': o['conf'],
            }
            for i, o in enumerate(observations)
        ],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate VGGT point maps accuracy")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["multiviewx"])
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--all_frames", action="store_true",
                        help="Evaluate all available frames")
    parser.add_argument("--conf_threshold", type=float, default=None,
                        help="Minimum confidence threshold for world_points")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Point Maps Validation (Phase 1 Gate)")
    print("=" * 60)

    output_dir = ROOT / "results" / "pointmap_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_frames:
        # Find all available VGGT prediction files
        pred_dir = ROOT / "results" / "vggt_predictions"
        frame_ids = sorted([
            int(p.stem.split('frame')[1])
            for p in pred_dir.glob(f"vggt_{args.dataset}_frame*.npz")
        ])
        print(f"Found {len(frame_ids)} frames: {frame_ids}")
    else:
        frame_ids = [args.frame_id]

    all_results = []
    for fid in frame_ids:
        result = evaluate_pointmaps(args.dataset, fid, args.conf_threshold)
        if result is None:
            continue

        # Save per-frame result
        json_path = output_dir / f"pointmap_eval_{args.dataset}_frame{fid}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {json_path}")
        all_results.append(result)

    # Multi-frame summary
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"Multi-Frame Summary ({len(all_results)} frames)")
        print(f"{'=' * 60}")

        xy_means = [r['summary']['error_xy']['mean'] for r in all_results]
        xy_medians = [r['summary']['error_xy']['median'] for r in all_results]
        consistency_means = [r['summary']['cross_view_consistency']['mean_deviation']
                             for r in all_results
                             if not np.isnan(r['summary']['cross_view_consistency']['mean_deviation'])]

        multi_summary = {
            'num_frames': len(all_results),
            'frame_ids': [r['summary']['frame_id'] for r in all_results],
            'error_xy_mean': {
                'across_frames_mean': float(np.mean(xy_means)),
                'across_frames_std': float(np.std(xy_means)),
                'across_frames_min': float(np.min(xy_means)),
                'across_frames_max': float(np.max(xy_means)),
            },
            'error_xy_median': {
                'across_frames_mean': float(np.mean(xy_medians)),
            },
            'cross_view_consistency': {
                'across_frames_mean': float(np.mean(consistency_means)) if consistency_means else float('nan'),
            },
            'per_frame_summaries': [r['summary'] for r in all_results],
        }

        json_path = output_dir / f"pointmap_multi_frame_{args.dataset}.json"
        with open(json_path, 'w') as f:
            json.dump(multi_summary, f, indent=2, ensure_ascii=False)
        print(f"Multi-frame summary saved: {json_path}")

        print(f"\n  XY error across frames: mean={np.mean(xy_means):.3f}m "
              f"(std={np.std(xy_means):.3f}m)")
        if consistency_means:
            print(f"  Cross-view consistency: mean={np.mean(consistency_means):.3f}m")

        # Overall gate
        overall_pass = np.mean(xy_means) < 1.0
        print(f"\n  Overall Gate (XY < 1.0m): {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
