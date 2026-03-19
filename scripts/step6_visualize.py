"""
阶段一 Step 6: 可视化评估结果

生成三类图表:
1. 相机位置对比鸟瞰图
2. 重投影可视化
3. 误差统计柱状图

用法:
    python scripts/step6_visualize.py --dataset wildtrack --frame_id 0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.coord_transform import (
    extrinsic_to_camera_center,
    align_poses_sim3,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(dataset, frame_id):
    """加载 GT 和 VGGT 数据"""
    # GT
    gt_data = np.load(str(ROOT / "results" / "gt_calibrations" / dataset / "gt_cameras.npz"))
    gt_cameras = {}
    cam_ids = set()
    for key in gt_data.files:
        cam_id = int(key.split('_')[0].replace('cam', ''))
        cam_ids.add(cam_id)
    for cam_id in sorted(cam_ids):
        gt_cameras[cam_id] = {
            'intrinsic': gt_data[f"cam{cam_id}_intrinsic"],
            'R': gt_data[f"cam{cam_id}_R"],
            't': gt_data[f"cam{cam_id}_t"],
            'center': gt_data[f"cam{cam_id}_center"],
        }

    # VGGT
    vggt_data = np.load(str(ROOT / "results" / "vggt_predictions" / f"vggt_{dataset}_frame{frame_id}.npz"))
    pred_cameras = {}
    N = vggt_data['extrinsics'].shape[0]
    for i in range(N):
        E = vggt_data['extrinsics'][i]
        K = vggt_data['intrinsics'][i]
        R = E[:3, :3]
        t = E[:3, 3]
        pred_cameras[i] = {
            'intrinsic': K,
            'R': R,
            't': t,
            'center': extrinsic_to_camera_center(R, t),
        }

    # 评估结果
    eval_path = ROOT / "results" / "evaluation" / f"evaluation_{dataset}_frame{frame_id}.json"
    eval_results = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

    return gt_cameras, pred_cameras, eval_results


def plot_camera_positions_birdseye(gt_cameras, pred_cameras, output_path):
    """
    图1: 鸟瞰图 - GT 相机位置 vs VGGT 估计位置（Sim3 对齐后）
    """
    cam_ids = sorted(gt_cameras.keys())

    gt_positions = np.array([gt_cameras[i]['center'] for i in cam_ids])
    pred_positions = np.array([pred_cameras[i]['center'] for i in cam_ids])

    # Sim(3) 对齐
    aligned_positions, _ = align_poses_sim3(pred_positions, gt_positions)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左图: XY 平面（鸟瞰图）---
    ax = axes[0]
    ax.set_title('Bird-eye View (XY Plane)', fontsize=14)

    for i, cam_id in enumerate(cam_ids):
        # GT
        ax.scatter(gt_positions[i, 0], gt_positions[i, 1],
                   c='blue', s=100, zorder=5, marker='o')
        ax.annotate(f'GT-{cam_id}', (gt_positions[i, 0], gt_positions[i, 1]),
                    fontsize=8, color='blue', textcoords="offset points", xytext=(5, 5))

        # VGGT（对齐后）
        ax.scatter(aligned_positions[i, 0], aligned_positions[i, 1],
                   c='red', s=100, zorder=5, marker='^')
        ax.annotate(f'V-{cam_id}', (aligned_positions[i, 0], aligned_positions[i, 1]),
                    fontsize=8, color='red', textcoords="offset points", xytext=(5, -10))

        # 连线
        ax.plot([gt_positions[i, 0], aligned_positions[i, 0]],
                [gt_positions[i, 1], aligned_positions[i, 1]],
                'k--', alpha=0.3, linewidth=1)

        # 相机朝向箭头（GT）
        R_gt = gt_cameras[cam_id]['R']
        forward_gt = -R_gt[2, :2]  # Z 轴方向在 XY 平面的投影
        forward_gt = forward_gt / (np.linalg.norm(forward_gt) + 1e-8) * 0.5
        ax.annotate('', xy=(gt_positions[i, 0] + forward_gt[0],
                            gt_positions[i, 1] + forward_gt[1]),
                    xytext=(gt_positions[i, 0], gt_positions[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(['GT', 'VGGT (aligned)'], loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- 右图: XZ 平面（侧视图）---
    ax = axes[1]
    ax.set_title('Side View (XZ Plane)', fontsize=14)

    for i, cam_id in enumerate(cam_ids):
        ax.scatter(gt_positions[i, 0], gt_positions[i, 2],
                   c='blue', s=100, zorder=5, marker='o')
        ax.scatter(aligned_positions[i, 0], aligned_positions[i, 2],
                   c='red', s=100, zorder=5, marker='^')
        ax.plot([gt_positions[i, 0], aligned_positions[i, 0]],
                [gt_positions[i, 2], aligned_positions[i, 2]],
                'k--', alpha=0.3, linewidth=1)

        ax.annotate(f'{cam_id}', (gt_positions[i, 0], gt_positions[i, 2]),
                    fontsize=8, color='blue', textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {output_path}")


def plot_error_bars(gt_cameras, pred_cameras, eval_results, output_path):
    """
    图2: 各相机的误差柱状图
    """
    cam_ids = sorted(gt_cameras.keys())

    gt_positions = np.array([gt_cameras[i]['center'] for i in cam_ids])
    pred_positions = np.array([pred_cameras[i]['center'] for i in cam_ids])
    aligned_positions, _ = align_poses_sim3(pred_positions, gt_positions)

    # 位置误差
    pos_errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)

    # 焦距误差
    focal_errors = []
    for cam_id in cam_ids:
        fx_gt = gt_cameras[cam_id]['intrinsic'][0, 0]
        fx_pred = pred_cameras[cam_id]['intrinsic'][0, 0]
        focal_errors.append(abs(fx_pred - fx_gt) / fx_gt * 100)

    # 重投影误差
    reproj_errors = []
    if eval_results and 'reprojection' in eval_results:
        for cam_id in cam_ids:
            reproj = eval_results['reprojection'].get(str(cam_id), {})
            reproj_errors.append(reproj.get('mean', 0))
    else:
        reproj_errors = [0] * len(cam_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(cam_ids))
    labels = [f'C{i}' for i in cam_ids]

    # 位置误差
    axes[0].bar(x, pos_errors, color='steelblue', alpha=0.8)
    axes[0].set_title('Position Error (m)', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Good (<0.2m)')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<0.5m)')
    axes[0].legend(fontsize=8)

    # 焦距误差
    axes[1].bar(x, focal_errors, color='coral', alpha=0.8)
    axes[1].set_title('Focal Length Error (%)', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Good (<5%)')
    axes[1].axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<15%)')
    axes[1].legend(fontsize=8)

    # 重投影误差
    axes[2].bar(x, reproj_errors, color='mediumpurple', alpha=0.8)
    axes[2].set_title('Reprojection Error (px)', fontsize=13)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Good (<10px)')
    axes[2].axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<30px)')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {output_path}")


def plot_intrinsic_comparison(gt_cameras, pred_cameras, output_path):
    """
    图3: 内参对比（焦距和主点）
    """
    cam_ids = sorted(gt_cameras.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(cam_ids))
    labels = [f'C{i}' for i in cam_ids]
    width = 0.35

    # 焦距对比
    fx_gt = [gt_cameras[i]['intrinsic'][0, 0] for i in cam_ids]
    fx_pred = [pred_cameras[i]['intrinsic'][0, 0] for i in cam_ids]

    axes[0].bar(x - width / 2, fx_gt, width, label='GT', color='steelblue', alpha=0.8)
    axes[0].bar(x + width / 2, fx_pred, width, label='VGGT', color='coral', alpha=0.8)
    axes[0].set_title('Focal Length (fx) Comparison', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].set_ylabel('Pixels')

    # 主点对比
    cx_gt = [gt_cameras[i]['intrinsic'][0, 2] for i in cam_ids]
    cx_pred = [pred_cameras[i]['intrinsic'][0, 2] for i in cam_ids]

    axes[1].bar(x - width / 2, cx_gt, width, label='GT', color='steelblue', alpha=0.8)
    axes[1].bar(x + width / 2, cx_pred, width, label='VGGT', color='coral', alpha=0.8)
    axes[1].set_title('Principal Point (cx) Comparison', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].set_ylabel('Pixels')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化评估结果")
    parser.add_argument("--dataset", type=str, default="wildtrack",
                        choices=["wildtrack", "multiviewx"])
    parser.add_argument("--frame_id", type=int, default=0)
    args = parser.parse_args()

    print(f"数据集: {args.dataset}, 帧: {args.frame_id}")
    print("=" * 50)

    gt_cameras, pred_cameras, eval_results = load_data(args.dataset, args.frame_id)

    n = min(len(gt_cameras), len(pred_cameras))
    gt_cameras = {i: gt_cameras[i] for i in range(n)}
    pred_cameras = {i: pred_cameras[i] for i in range(n)}

    output_dir = ROOT / "results" / "figures" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n生成图表:")

    plot_camera_positions_birdseye(
        gt_cameras, pred_cameras,
        output_dir / f"camera_positions_frame{args.frame_id}.png"
    )

    plot_error_bars(
        gt_cameras, pred_cameras, eval_results,
        output_dir / f"error_bars_frame{args.frame_id}.png"
    )

    plot_intrinsic_comparison(
        gt_cameras, pred_cameras,
        output_dir / f"intrinsic_comparison_frame{args.frame_id}.png"
    )

    print(f"\n所有图表已保存到: {output_dir}/")


if __name__ == "__main__":
    main()
