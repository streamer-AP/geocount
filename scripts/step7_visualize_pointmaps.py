"""
Phase 1 Step 7: Point Maps 可视化

生成:
1. BEV 散点图: GT 位置 vs VGGT 投影 (Sim(3) 对齐后)
2. 误差分布直方图
3. 逐相机误差对比
4. 跨视角一致性可视化

用法:
    python scripts/step7_visualize_pointmaps.py --dataset multiviewx --frame_id 0
    python scripts/step7_visualize_pointmaps.py --dataset multiviewx --all_frames
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATASET_PARAMS = {
    'multiviewx': {
        'map_width': 25,
        'map_height': 16,
        'num_cam': 6,
    },
}


def load_eval_result(dataset, frame_id):
    """Load pointmap evaluation result JSON."""
    path = ROOT / "results" / "pointmap_validation" / f"pointmap_eval_{dataset}_frame{frame_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation result not found: {path}\n"
            f"Run: python scripts/step7_validate_pointmaps.py --dataset {dataset} --frame_id {frame_id}"
        )
    with open(path) as f:
        return json.load(f)


def plot_bev_scatter(result, dataset, frame_id, output_path):
    """BEV scatter plot: GT positions vs VGGT aligned projections."""
    params = DATASET_PARAMS[dataset]
    obs = result['observations']

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: All cameras overlaid ---
    ax = axes[0]
    ax.set_title(f'BEV: GT vs VGGT Projections (Frame {frame_id})', fontsize=13)

    # Scene boundary
    ax.add_patch(Rectangle((0, 0), params['map_width'], params['map_height'],
                            fill=False, edgecolor='black', linewidth=2, linestyle='--'))

    colors = plt.cm.tab10(np.linspace(0, 1, params['num_cam']))
    cam_labels_added = set()

    for o in obs:
        cam_id = o['cam_id']
        gt = o['gt_xy']
        aligned = o['aligned_3d'][:2]

        label_gt = f'GT' if 'gt' not in cam_labels_added else None
        label_vggt = f'VGGT cam{cam_id}' if cam_id not in cam_labels_added else None
        cam_labels_added.add(cam_id)
        cam_labels_added.add('gt')

        ax.scatter(gt[0], gt[1], c='black', s=30, marker='o', alpha=0.5,
                   label=label_gt, zorder=3)
        ax.scatter(aligned[0], aligned[1], c=[colors[cam_id]], s=20, marker='^',
                   alpha=0.6, label=label_vggt, zorder=2)
        # Connect GT to projection
        ax.plot([gt[0], aligned[0]], [gt[1], aligned[1]],
                c=colors[cam_id], alpha=0.15, linewidth=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(-1, params['map_width'] + 1)
    ax.set_ylim(-1, params['map_height'] + 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Right: Error heatmap on BEV grid ---
    ax = axes[1]
    ax.set_title(f'Mean XY Error per BEV Cell (0.5m grid)', fontsize=13)

    grid_dx = 0.5
    grid_nx = int(params['map_width'] / grid_dx)
    grid_ny = int(params['map_height'] / grid_dx)
    error_sum = np.zeros((grid_ny, grid_nx))
    error_count = np.zeros((grid_ny, grid_nx))

    for o in obs:
        gt = o['gt_xy']
        gx = int(gt[0] / grid_dx)
        gy = int(gt[1] / grid_dx)
        if 0 <= gx < grid_nx and 0 <= gy < grid_ny:
            error_sum[gy, gx] += o['error_xy']
            error_count[gy, gx] += 1

    error_mean = np.full((grid_ny, grid_nx), np.nan)
    mask = error_count > 0
    error_mean[mask] = error_sum[mask] / error_count[mask]

    im = ax.imshow(error_mean, origin='lower', cmap='RdYlGn_r',
                   extent=[0, params['map_width'], 0, params['map_height']],
                   vmin=0, vmax=2.0, aspect='equal')
    plt.colorbar(im, ax=ax, label='Mean XY Error (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_error_distribution(result, dataset, frame_id, output_path):
    """Error distribution histograms."""
    obs = result['observations']
    params = DATASET_PARAMS[dataset]
    errors_xy = [o['error_xy'] for o in obs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left: Overall error histogram ---
    ax = axes[0]
    ax.hist(errors_xy, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(x=np.mean(errors_xy), color='red', linestyle='--',
               label=f'Mean: {np.mean(errors_xy):.3f}m')
    ax.axvline(x=np.median(errors_xy), color='orange', linestyle='--',
               label=f'Median: {np.median(errors_xy):.3f}m')
    ax.axvline(x=1.0, color='green', linestyle='-', alpha=0.5,
               label='Pass threshold: 1.0m')
    ax.set_xlabel('XY Error (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (n={len(errors_xy)})', fontsize=13)
    ax.legend(fontsize=8)

    # --- Middle: Per-camera box plot ---
    ax = axes[1]
    per_cam_errors = {}
    for o in obs:
        cam_id = o['cam_id']
        if cam_id not in per_cam_errors:
            per_cam_errors[cam_id] = []
        per_cam_errors[cam_id].append(o['error_xy'])

    cam_ids = sorted(per_cam_errors.keys())
    box_data = [per_cam_errors[c] for c in cam_ids]
    bp = ax.boxplot(box_data, tick_labels=[f'C{c}' for c in cam_ids], patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(cam_ids)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Pass: 1.0m')
    ax.set_ylabel('XY Error (m)')
    ax.set_title('Per-Camera Error Distribution', fontsize=13)
    ax.legend(fontsize=8)

    # --- Right: Cross-view consistency ---
    ax = axes[2]
    consistency = result.get('per_person_consistency', [])
    if consistency:
        devs = [p['mean_deviation'] for p in consistency]
        ax.hist(devs, bins=20, color='coral', alpha=0.8, edgecolor='white')
        ax.axvline(x=np.mean(devs), color='red', linestyle='--',
                   label=f'Mean: {np.mean(devs):.3f}m')
        ax.axvline(x=1.0, color='green', linestyle='-', alpha=0.5,
                   label='Pass: 1.0m')
        ax.set_xlabel('Mean Deviation from Centroid (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'Cross-View Consistency (n={len(devs)})', fontsize=13)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No multi-view data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Cross-View Consistency', fontsize=13)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_camera_detail(result, dataset, frame_id, output_path):
    """Per-camera BEV scatter (one subplot per camera)."""
    obs = result['observations']
    params = DATASET_PARAMS[dataset]
    num_cam = params['num_cam']

    ncols = 3
    nrows = (num_cam + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for cam_id in range(num_cam):
        ax = axes[cam_id]
        cam_obs = [o for o in obs if o['cam_id'] == cam_id]

        ax.add_patch(Rectangle((0, 0), params['map_width'], params['map_height'],
                                fill=False, edgecolor='black', linewidth=1.5, linestyle='--'))

        if cam_obs:
            gt_x = [o['gt_xy'][0] for o in cam_obs]
            gt_y = [o['gt_xy'][1] for o in cam_obs]
            al_x = [o['aligned_3d'][0] for o in cam_obs]
            al_y = [o['aligned_3d'][1] for o in cam_obs]
            errs = [o['error_xy'] for o in cam_obs]

            ax.scatter(gt_x, gt_y, c='blue', s=40, marker='o', alpha=0.7, label='GT')
            sc = ax.scatter(al_x, al_y, c=errs, s=40, marker='^', alpha=0.7,
                            cmap='RdYlGn_r', vmin=0, vmax=2.0, label='VGGT')
            for gx, gy, ax_, ay_ in zip(gt_x, gt_y, al_x, al_y):
                ax.plot([gx, ax_], [gy, ay_], 'k-', alpha=0.15, linewidth=0.5)

            plt.colorbar(sc, ax=ax, label='Error (m)', shrink=0.8)
            mean_err = np.mean(errs)
            ax.set_title(f'Camera {cam_id} (n={len(cam_obs)}, mean={mean_err:.2f}m)', fontsize=11)
        else:
            ax.set_title(f'Camera {cam_id} (no data)', fontsize=11)

        ax.set_xlim(-1, params['map_width'] + 1)
        ax.set_ylim(-1, params['map_height'] + 1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(num_cam, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Per-Camera Point Map Projections (Frame {frame_id})', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_multi_frame_summary(dataset, output_path):
    """Multi-frame summary plot across all evaluated frames."""
    summary_path = ROOT / "results" / "pointmap_validation" / f"pointmap_multi_frame_{dataset}.json"
    if not summary_path.exists():
        print("  Multi-frame summary not found, skipping")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    per_frame = summary['per_frame_summaries']
    frame_ids = [s['frame_id'] for s in per_frame]
    xy_means = [s['error_xy']['mean'] for s in per_frame]
    xy_medians = [s['error_xy']['median'] for s in per_frame]
    consistency = [s['cross_view_consistency']['mean_deviation'] for s in per_frame]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # XY error across frames
    ax = axes[0]
    ax.bar(frame_ids, xy_means, alpha=0.8, color='steelblue', label='Mean')
    ax.plot(frame_ids, xy_medians, 'ro-', markersize=5, label='Median')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Pass: 1.0m')
    ax.set_xlabel('Frame ID')
    ax.set_ylabel('XY Error (m)')
    ax.set_title('Ground Projection Error Across Frames', fontsize=13)
    ax.legend(fontsize=8)
    ax.set_xticks(frame_ids)

    # Cross-view consistency
    ax = axes[1]
    valid_cons = [(f, c) for f, c in zip(frame_ids, consistency) if not np.isnan(c)]
    if valid_cons:
        ax.bar([f for f, _ in valid_cons], [c for _, c in valid_cons],
               alpha=0.8, color='coral')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Pass: 1.0m')
    ax.set_xlabel('Frame ID')
    ax.set_ylabel('Mean Deviation (m)')
    ax.set_title('Cross-View Consistency', fontsize=13)
    ax.legend(fontsize=8)
    ax.set_xticks(frame_ids)

    # Sim3 scale
    ax = axes[2]
    scales = [s['sim3_params']['scale'] for s in per_frame]
    ax.plot(frame_ids, scales, 'go-', markersize=6)
    ax.set_xlabel('Frame ID')
    ax.set_ylabel('Sim(3) Scale')
    ax.set_title('Sim(3) Scale Across Frames', fontsize=13)
    ax.set_xticks(frame_ids)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize point map validation results")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["multiviewx"])
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--all_frames", action="store_true")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print("Point Maps Visualization")
    print("=" * 60)

    output_dir = ROOT / "results" / "figures" / "pointmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = ROOT / "results" / "pointmap_validation"

    if args.all_frames:
        frame_ids = sorted([
            int(p.stem.split('frame')[1])
            for p in eval_dir.glob(f"pointmap_eval_{args.dataset}_frame*.json")
        ])
        print(f"Found {len(frame_ids)} evaluated frames: {frame_ids}")
    else:
        frame_ids = [args.frame_id]

    for fid in frame_ids:
        print(f"\n--- Frame {fid} ---")
        result = load_eval_result(args.dataset, fid)

        plot_bev_scatter(result, args.dataset, fid,
                         output_dir / f"bev_scatter_{args.dataset}_frame{fid}.png")
        plot_error_distribution(result, args.dataset, fid,
                                output_dir / f"error_dist_{args.dataset}_frame{fid}.png")
        plot_per_camera_detail(result, args.dataset, fid,
                               output_dir / f"per_camera_{args.dataset}_frame{fid}.png")

    # Multi-frame summary
    if args.all_frames and len(frame_ids) > 1:
        plot_multi_frame_summary(args.dataset,
                                 output_dir / f"multi_frame_{args.dataset}.png")

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
