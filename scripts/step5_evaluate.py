"""
阶段一 Step 5: 评估 VGGT 估计的相机参数精度

将 VGGT 输出与 GT 标定参数对比，计算各项误差指标。

用法:
    python scripts/step5_evaluate.py --dataset wildtrack --frame_id 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.coord_transform import (
    extrinsic_to_camera_center,
    align_poses_sim3,
)
from utils.metrics import (
    compute_relative_metrics,
    compute_absolute_metrics,
    compute_reprojection_error,
    summarize_metrics,
)


def load_gt_cameras(dataset):
    """加载 GT 相机参数"""
    npz_path = ROOT / "results" / "gt_calibrations" / dataset / "gt_cameras.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"GT 标定文件不存在: {npz_path}\n"
            f"请先运行: python scripts/step3_parse_gt_calib.py --dataset {dataset}"
        )

    data = np.load(str(npz_path))
    cameras = {}
    cam_ids = set()
    for key in data.files:
        cam_id = int(key.split('_')[0].replace('cam', ''))
        cam_ids.add(cam_id)

    for cam_id in sorted(cam_ids):
        cameras[cam_id] = {
            'intrinsic': data[f"cam{cam_id}_intrinsic"],
            'R': data[f"cam{cam_id}_R"],
            't': data[f"cam{cam_id}_t"],
            'center': data[f"cam{cam_id}_center"],
        }
    return cameras


def load_vggt_predictions(dataset, frame_id):
    """加载 VGGT 推理结果"""
    npz_path = ROOT / "results" / "vggt_predictions" / f"vggt_{dataset}_frame{frame_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"VGGT 预测文件不存在: {npz_path}\n"
            f"请先运行: python scripts/step4_run_vggt.py --dataset {dataset} --frame_id {frame_id}"
        )

    data = np.load(str(npz_path))
    return {
        'extrinsics': data['extrinsics'],  # (N, 3, 4)
        'intrinsics': data['intrinsics'],  # (N, 3, 3)，基于 resize 后分辨率
        'resized_hw': data['resized_hw'] if 'resized_hw' in data else None,
    }


# 各数据集原始分辨率
DATASET_ORIG_RESOLUTION = {
    'wildtrack':  (1080, 1920),
    'multiviewx': (1080, 1920),
}


def rescale_intrinsics(K, orig_hw, resized_hw):
    """
    将 VGGT 输出的内参从 resize 后分辨率换算回原始分辨率。

    VGGT 将图像 resize 到 (resized_h, resized_w)，输出的焦距和主点
    都是基于该分辨率的，需要乘以对应的缩放比例才能与 GT 比较。

    Parameters:
        K:          (3, 3) VGGT 输出的内参矩阵
        orig_hw:    (H, W) 原始图像分辨率
        resized_hw: (H, W) VGGT 处理时的分辨率

    Returns:
        K_scaled: (3, 3) 换算到原始分辨率的内参矩阵
    """
    orig_h, orig_w = orig_hw
    res_h, res_w = resized_hw
    scale_x = orig_w / res_w
    scale_y = orig_h / res_h

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x   # fx
    K_scaled[1, 1] *= scale_y   # fy
    K_scaled[0, 2] *= scale_x   # cx
    K_scaled[1, 2] *= scale_y   # cy
    return K_scaled


def build_pred_cameras(vggt_data, dataset, resized_hw=None):
    """
    将 VGGT 输出转换为与 GT 相同的 dict 格式。
    自动将内参从 resize 后分辨率换算回原始分辨率。
    """
    cameras = {}
    N = vggt_data['extrinsics'].shape[0]

    orig_hw = DATASET_ORIG_RESOLUTION.get(dataset)
    if resized_hw is None:
        resized_hw = vggt_data.get('resized_hw')

    for i in range(N):
        E = vggt_data['extrinsics'][i]
        K = vggt_data['intrinsics'][i].copy()
        R = E[:3, :3]
        t = E[:3, 3]
        center = extrinsic_to_camera_center(R, t)

        if orig_hw is not None and resized_hw is not None:
            K = rescale_intrinsics(K, orig_hw, resized_hw)

        cameras[i] = {
            'intrinsic': K,
            'R': R,
            't': t,
            'center': center,
        }
    return cameras


def generate_ground_plane_points(n_points=100, x_range=(-5, 5), y_range=(-5, 5)):
    """
    在地面平面上生成均匀采样的 3D 点，用于重投影误差计算。
    假设地面平面为 z=0。
    """
    x = np.linspace(x_range[0], x_range[1], int(np.sqrt(n_points)))
    y = np.linspace(y_range[0], y_range[1], int(np.sqrt(n_points)))
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    return points


def evaluate(gt_cameras, pred_cameras, height_constraint=None):
    """完整的评估流程

    Parameters:
        gt_cameras:        dict, cam_id -> {R, t, intrinsic, center}
        pred_cameras:      dict, cam_id -> {R, t, intrinsic, center}
        height_constraint: float 或 None。
            若不为 None，则在 Sim(3) 对齐后将所有预测相机的 z 坐标
            强制设为该值（等高约束，利用监控相机安装高度已知的先验）。
            传入 0.0 表示自动使用对齐后 z 坐标的均值。
    """
    cam_ids = sorted(gt_cameras.keys())
    n_cams = len(cam_ids)

    print(f"\n共 {n_cams} 个相机")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. Sim(3) 对齐
    # -------------------------------------------------------
    print("\n[1/4] Sim(3) 坐标系对齐...")

    gt_positions = np.array([gt_cameras[i]['center'] for i in cam_ids])
    pred_positions = np.array([pred_cameras[i]['center'] for i in cam_ids])

    print(f"  GT 位置范围:   {gt_positions.min(axis=0)} ~ {gt_positions.max(axis=0)}")
    print(f"  Pred 位置范围: {pred_positions.min(axis=0)} ~ {pred_positions.max(axis=0)}")

    aligned_positions, sim3_params = align_poses_sim3(pred_positions, gt_positions)
    print(f"  对齐参数: scale={sim3_params['s']:.4f}")
    print(f"  对齐后位置范围: {aligned_positions.min(axis=0)} ~ {aligned_positions.max(axis=0)}")

    # -------------------------------------------------------
    # 等高约束（可选）：监控相机通常安装在同一高度
    # -------------------------------------------------------
    if height_constraint is not None:
        h = aligned_positions[:, 2].mean() if height_constraint == 0.0 else height_constraint
        print(f"\n  [等高约束] 将所有相机 z 坐标强制设为 {h:.3f}m"
              f"（原均值 {aligned_positions[:, 2].mean():.3f}m）")
        aligned_positions = aligned_positions.copy()
        aligned_positions[:, 2] = h

    # -------------------------------------------------------
    # 2. 绝对指标
    # -------------------------------------------------------
    print("\n[2/4] 计算绝对指标...")

    abs_results = compute_absolute_metrics(pred_cameras, gt_cameras, aligned_positions)

    print(f"\n  {'相机':>4} | {'位置误差(m)':>10} | {'焦距误差(%)':>10} | "
          f"{'fx_pred':>8} {'fx_gt':>8} | {'fy_pred':>8} {'fy_gt':>8}")
    print("  " + "-" * 80)
    for cam_id in cam_ids:
        r = abs_results[cam_id]
        print(f"  {cam_id:>4} | {r['position_error_m']:>10.4f} | {r['focal_error_pct']:>10.2f} | "
              f"{r['fx_pred']:>8.1f} {r['fx_gt']:>8.1f} | {r['fy_pred']:>8.1f} {r['fy_gt']:>8.1f}")

    # -------------------------------------------------------
    # 3. 相对指标（不受坐标系影响）
    # -------------------------------------------------------
    print("\n[3/4] 计算相对指标...")

    rel_results = compute_relative_metrics(pred_cameras, gt_cameras)

    print(f"\n  {'相机对':>8} | {'相对旋转误差(度)':>16} | {'相对平移角度误差(度)':>20}")
    print("  " + "-" * 55)
    for (i, j), r in sorted(rel_results.items()):
        print(f"  ({i},{j}){' ':>3} | {r['relative_rotation_error_deg']:>16.2f} | "
              f"{r['relative_translation_angle_deg']:>20.2f}")

    # -------------------------------------------------------
    # 4. 重投影误差
    # -------------------------------------------------------
    print("\n[4/4] 计算重投影误差...")

    # 构造对齐后的 pred 相机参数
    R_align = sim3_params['R']
    s = sim3_params['s']
    t_align = sim3_params['t']

    aligned_pred_cameras = {}
    for idx, cam_id in enumerate(cam_ids):
        pred = pred_cameras[cam_id]
        # 对齐旋转: R_aligned = R_pred @ R_align^T
        R_aligned = pred['R'] @ R_align.T
        # 对齐平移: 从对齐后的 center 反算 t
        center_aligned = aligned_positions[idx]
        t_aligned = -R_aligned @ center_aligned

        aligned_pred_cameras[cam_id] = {
            'intrinsic': pred['intrinsic'],
            'R': R_aligned,
            't': t_aligned,
        }

    # 在地面平面上采样 3D 点
    # 用 dataset 实际场地尺寸（而非仅由相机位置推算），确保覆盖完整场地
    DATASET_GROUND_RANGE = {
        'multiviewx': (0, 25, 0, 16),   # x: 0-25m, y: 0-16m
        'wildtrack':  (0, 12, 0, 36),   # x: 0-12m, y: 0-36m (约)
    }
    orig_hw = DATASET_ORIG_RESOLUTION.get(args.dataset, (1080, 1920))
    image_wh = (orig_hw[1], orig_hw[0])  # (W, H)

    if args.dataset in DATASET_GROUND_RANGE:
        x0, x1, y0, y1 = DATASET_GROUND_RANGE[args.dataset]
    else:
        gt_centers_xy = gt_positions[:, :2]
        center_of_scene = gt_centers_xy.mean(axis=0)
        scene_radius = np.linalg.norm(gt_centers_xy - center_of_scene, axis=1).max()
        x0 = center_of_scene[0] - scene_radius
        x1 = center_of_scene[0] + scene_radius
        y0 = center_of_scene[1] - scene_radius
        y1 = center_of_scene[1] + scene_radius

    ground_points = generate_ground_plane_points(
        n_points=400,
        x_range=(x0, x1),
        y_range=(y0, y1),
    )

    print(f"\n  地面采样点: {ground_points.shape[0]} 个")
    print(f"  {'相机':>4} | {'平均重投影误差(px)':>18} | {'中位数(px)':>10} | {'最大(px)':>8}")
    print("  " + "-" * 55)

    reproj_errors_all = {}
    for cam_id in cam_ids:
        mean_err, errors = compute_reprojection_error(
            ground_points,
            aligned_pred_cameras[cam_id],
            gt_cameras[cam_id],
            image_wh=image_wh,
        )
        # 过滤掉两侧任一深度接近零（无效）的点
        valid = np.isfinite(errors)
        if valid.sum() > 0:
            errors_valid = errors[valid]
            mean_err = errors_valid.mean()
            median_err = np.median(errors_valid)
            max_err = errors_valid.max()
        else:
            mean_err = median_err = max_err = float('nan')

        reproj_errors_all[cam_id] = {
            'mean': mean_err,
            'median': median_err,
            'max': max_err,
        }
        print(f"  {cam_id:>4} | {mean_err:>18.2f} | {median_err:>10.2f} | {max_err:>8.2f}")

    # -------------------------------------------------------
    # 汇总
    # -------------------------------------------------------
    summary = summarize_metrics(abs_results, rel_results)

    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    for metric_name, vals in summary.items():
        print(f"\n  {metric_name}:")
        for stat_name, val in vals.items():
            print(f"    {stat_name}: {val:.4f}")

    # 平均重投影误差
    reproj_means = [v['mean'] for v in reproj_errors_all.values() if not np.isnan(v['mean'])]
    if reproj_means:
        print(f"\n  reprojection_error_px:")
        print(f"    mean: {np.mean(reproj_means):.2f}")
        print(f"    max:  {np.max(reproj_means):.2f}")

    # -------------------------------------------------------
    # 可行性判断
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("可行性判断")
    print("=" * 60)

    rre_mean = summary['relative_rotation_error_deg']['mean']
    pos_mean = summary['position_error_m']['mean']
    reproj_mean = np.mean(reproj_means) if reproj_means else float('nan')
    focal_mean = summary['focal_error_pct']['mean']

    def grade(val, thresholds, labels=("优秀", "可用", "需改进")):
        if val < thresholds[0]:
            return labels[0]
        elif val < thresholds[1]:
            return labels[1]
        return labels[2]

    print(f"  相对旋转误差:  {rre_mean:.2f}度  -> {grade(rre_mean, [2, 5])}")
    print(f"  位置误差:      {pos_mean:.4f}m  -> {grade(pos_mean, [0.2, 0.5])}")
    print(f"  重投影误差:    {reproj_mean:.2f}px -> {grade(reproj_mean, [10, 30])}")
    print(f"  内参误差:      {focal_mean:.2f}%  -> {grade(focal_mean, [5, 15])}")

    # 保存结果
    output_dir = ROOT / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_to_save = {
        'summary': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in summary.items()},
        'reprojection': {str(k): {sk: float(sv) for sk, sv in v.items()}
                         for k, v in reproj_errors_all.items()},
        'sim3_params': {
            'scale': float(sim3_params['s']),
            'R': sim3_params['R'].tolist(),
            't': sim3_params['t'].tolist(),
        },
    }

    json_path = output_dir / f"evaluation_{args.dataset}_frame{args.frame_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {json_path}")

    return results_to_save


def main():
    global args
    parser = argparse.ArgumentParser(description="评估 VGGT 相机参数精度")
    parser.add_argument("--dataset", type=str, default="wildtrack",
                        choices=["wildtrack", "multiviewx"])
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--height_constraint", type=float, default=None,
                        help="等高约束：将所有相机 z 坐标强制对齐到指定高度（米）。"
                             "传入 0 则自动取对齐后 z 均值。不传则不启用。"
                             "监控相机通常安装在同一高度（如 2.5m），该约束可显著降低位置误差。")
    args = parser.parse_args()

    print(f"数据集: {args.dataset}")
    print(f"帧: {args.frame_id}")
    print("=" * 60)

    # 加载数据
    gt_cameras = load_gt_cameras(args.dataset)
    vggt_data = load_vggt_predictions(args.dataset, args.frame_id)

    # 从 npz 中读取 resized_hw（由 step4 保存），若无则从 extrinsics 推断不到，需手动指定
    resized_hw = tuple(vggt_data['resized_hw'].astype(int)) if vggt_data.get('resized_hw') is not None else None
    if resized_hw is None:
        print("[警告] 未找到 resized_hw，内参换算将跳过（请重新运行 step4 以保存该信息）")

    pred_cameras = build_pred_cameras(vggt_data, args.dataset, resized_hw)

    # 检查相机数量一致
    if len(gt_cameras) != len(pred_cameras):
        print(f"[警告] GT 相机数 ({len(gt_cameras)}) != VGGT 相机数 ({len(pred_cameras)})")
        n = min(len(gt_cameras), len(pred_cameras))
        gt_cameras = {i: gt_cameras[i] for i in range(n)}
        pred_cameras = {i: pred_cameras[i] for i in range(n)}

    # 评估
    evaluate(gt_cameras, pred_cameras, height_constraint=args.height_constraint)


if __name__ == "__main__":
    main()
