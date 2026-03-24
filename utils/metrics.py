"""评估指标计算工具"""

import numpy as np
from itertools import combinations
from .coord_transform import (
    rotation_error,
    relative_rotation_error,
    relative_translation_angle_error,
    extrinsic_to_camera_center,
)


def compute_absolute_metrics(pred_cameras, gt_cameras, aligned_positions):
    """
    计算绝对指标（需要先做 Sim(3) 对齐）。

    Parameters:
        pred_cameras: dict, cam_id -> {'R': (3,3), 't': (3,), 'intrinsic': (3,3)}
        gt_cameras:   dict, cam_id -> {'R': (3,3), 't': (3,), 'intrinsic': (3,3)}
        aligned_positions: (N, 3) 对齐后的预测相机位置

    Returns:
        dict: 每个相机的指标
    """
    cam_ids = sorted(gt_cameras.keys())
    results = {}

    for idx, cam_id in enumerate(cam_ids):
        gt = gt_cameras[cam_id]
        pred = pred_cameras[cam_id]

        # 相机位置误差（对齐后）
        gt_center = extrinsic_to_camera_center(gt['R'], gt['t'])
        pos_error = np.linalg.norm(aligned_positions[idx] - gt_center)

        # 内参误差
        K_pred = pred['intrinsic']
        K_gt = gt['intrinsic']
        fx_error = abs(K_pred[0, 0] - K_gt[0, 0]) / K_gt[0, 0] * 100
        fy_error = abs(K_pred[1, 1] - K_gt[1, 1]) / K_gt[1, 1] * 100
        focal_error = (fx_error + fy_error) / 2

        results[cam_id] = {
            'position_error_m': pos_error,
            'focal_error_pct': focal_error,
            'fx_pred': K_pred[0, 0],
            'fy_pred': K_pred[1, 1],
            'fx_gt': K_gt[0, 0],
            'fy_gt': K_gt[1, 1],
        }

    return results


def compute_relative_metrics(pred_cameras, gt_cameras):
    """
    计算相对指标（不受坐标系/尺度影响，最可靠的指标）。

    Parameters:
        pred_cameras: dict, cam_id -> {'R': (3,3), 't': (3,)}
        gt_cameras:   dict, cam_id -> {'R': (3,3), 't': (3,)}

    Returns:
        dict: 每对相机的相对旋转误差和相对平移角度误差
    """
    cam_ids = sorted(gt_cameras.keys())
    results = {}

    # 提取相机中心
    pred_centers = {}
    gt_centers = {}
    for cam_id in cam_ids:
        pred_centers[cam_id] = extrinsic_to_camera_center(
            pred_cameras[cam_id]['R'], pred_cameras[cam_id]['t']
        )
        gt_centers[cam_id] = extrinsic_to_camera_center(
            gt_cameras[cam_id]['R'], gt_cameras[cam_id]['t']
        )

    for i, j in combinations(cam_ids, 2):
        rre = relative_rotation_error(
            pred_cameras[i]['R'], pred_cameras[j]['R'],
            gt_cameras[i]['R'], gt_cameras[j]['R'],
        )
        rte = relative_translation_angle_error(
            pred_centers[i], pred_centers[j],
            gt_centers[i], gt_centers[j],
        )
        results[(i, j)] = {
            'relative_rotation_error_deg': rre,
            'relative_translation_angle_deg': rte,
        }

    return results


def compute_reprojection_error(gt_3d_points, cam_pred, cam_gt,
                               image_wh=None):
    """
    计算重投影误差。
    用 GT 参数和 pred 参数分别将 3D 点投影到图像，计算像素距离。

    注意：GT 外参允许负深度（如 MultiviewX/Wildtrack 使用相机朝-z约定），
    投影公式直接用 x/z, y/z，不过滤 z 的符号。
    只过滤 |z| 极小（投影无意义）或 GT 投影点在图像范围外的点。

    Parameters:
        gt_3d_points: (M, 3) 世界坐标系中的 3D 点
        cam_pred:     dict with 'R', 't', 'intrinsic' (对齐后)
        cam_gt:       dict with 'R', 't', 'intrinsic'
        image_wh:     (W, H) 图像分辨率，用于过滤 GT 在图像外的点；
                      为 None 时不过滤

    Returns:
        mean_error: 平均重投影误差（像素），无有效点时为 inf
        errors:     (M,) 每个点的误差（无效点为 inf）
    """
    pts = gt_3d_points.T  # (3, M)

    # GT 投影（允许负深度）
    p_cam_gt = cam_gt['R'] @ pts + cam_gt['t'].reshape(3, 1)
    valid_gt = np.abs(p_cam_gt[2]) > 1e-3
    proj_gt = np.full((2, pts.shape[1]), np.inf)
    proj_gt[:, valid_gt] = (
        cam_gt['intrinsic'] @ (p_cam_gt[:, valid_gt] / p_cam_gt[2:3, valid_gt])
    )[:2]

    # 过滤 GT 投影在图像范围外的点
    in_frame = np.ones(pts.shape[1], dtype=bool)
    if image_wh is not None:
        W, H = image_wh
        in_frame = (
            (proj_gt[0] >= 0) & (proj_gt[0] < W) &
            (proj_gt[1] >= 0) & (proj_gt[1] < H)
        )

    # 预测参数投影（对齐后，期望 z > 0）
    p_cam_pred = cam_pred['R'] @ pts + cam_pred['t'].reshape(3, 1)
    valid_pred = np.abs(p_cam_pred[2]) > 1e-3
    proj_pred = np.full((2, pts.shape[1]), np.inf)
    proj_pred[:, valid_pred] = (
        cam_pred['intrinsic'] @ (p_cam_pred[:, valid_pred] / p_cam_pred[2:3, valid_pred])
    )[:2]

    # 像素误差（GT有效 & GT在图像内 & pred有效）
    valid = valid_gt & in_frame & valid_pred
    errors = np.full(pts.shape[1], np.inf)
    errors[valid] = np.linalg.norm(proj_pred[:, valid] - proj_gt[:, valid], axis=0)
    mean_error = errors[valid].mean() if valid.any() else np.inf
    return mean_error, errors


def summarize_metrics(absolute_results, relative_results):
    """汇总所有指标，输出摘要。"""
    # 绝对指标摘要
    pos_errors = [v['position_error_m'] for v in absolute_results.values()]
    focal_errors = [v['focal_error_pct'] for v in absolute_results.values()]

    # 相对指标摘要
    rre_list = [v['relative_rotation_error_deg'] for v in relative_results.values()]
    rte_list = [v['relative_translation_angle_deg'] for v in relative_results.values()
                if not np.isnan(v['relative_translation_angle_deg'])]

    summary = {
        'position_error_m': {
            'mean': np.mean(pos_errors),
            'median': np.median(pos_errors),
            'max': np.max(pos_errors),
        },
        'focal_error_pct': {
            'mean': np.mean(focal_errors),
            'median': np.median(focal_errors),
            'max': np.max(focal_errors),
        },
        'relative_rotation_error_deg': {
            'mean': np.mean(rre_list),
            'median': np.median(rre_list),
            'max': np.max(rre_list),
        },
        'relative_translation_angle_deg': {
            'mean': np.mean(rte_list) if rte_list else float('nan'),
            'median': np.median(rte_list) if rte_list else float('nan'),
            'max': np.max(rte_list) if rte_list else float('nan'),
        },
    }

    return summary
