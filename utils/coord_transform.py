"""坐标系转换与位姿对齐工具"""

import numpy as np
from scipy.spatial.transform import Rotation


def extrinsic_to_camera_center(R, t):
    """
    从外参 [R|t] 提取相机在世界坐标系中的位置。
    外参定义: x_cam = R @ x_world + t
    相机中心: C = -R^T @ t
    """
    return -R.T @ t.reshape(3)


def camera_center_to_extrinsic(R, C):
    """
    从旋转矩阵和相机中心构造外参。
    t = -R @ C
    """
    t = -R @ C.reshape(3)
    return np.hstack([R, t.reshape(3, 1)])


def align_poses_sim3(pred_positions, gt_positions):
    """
    Sim(3) 对齐（Umeyama 算法）。
    找到最优的 s, R, t 使得 gt ≈ s * R @ pred + t

    Parameters:
        pred_positions: (N, 3) 预测的相机位置
        gt_positions:   (N, 3) GT 相机位置

    Returns:
        aligned: (N, 3) 对齐后的预测位置
        params:  dict with 's', 'R', 't'
    """
    assert pred_positions.shape == gt_positions.shape
    n = pred_positions.shape[0]

    # 中心化
    pred_mean = pred_positions.mean(axis=0)
    gt_mean = gt_positions.mean(axis=0)
    pred_c = pred_positions - pred_mean
    gt_c = gt_positions - gt_mean

    # 尺度
    pred_var = (pred_c ** 2).sum() / n
    gt_var = (gt_c ** 2).sum() / n

    if pred_var < 1e-10:
        raise ValueError("预测位置方差接近零，无法对齐")

    # 协方差矩阵
    H = pred_c.T @ gt_c / n

    # SVD
    U, D, Vt = np.linalg.svd(H)

    # 处理反射
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = Vt.T @ S @ U.T
    s = np.trace(np.diag(D) @ S) / pred_var
    t = gt_mean - s * R @ pred_mean

    aligned = s * (R @ pred_positions.T).T + t

    return aligned, {'R': R, 't': t, 's': s}


def align_rotations_sim3(pred_rotations, gt_rotations, sim3_params):
    """
    用 Sim(3) 参数对齐旋转矩阵。
    R_aligned = R_gt_frame @ R_pred
    其中 R_gt_frame = sim3_params['R']

    Parameters:
        pred_rotations: list of (3, 3) 预测的旋转矩阵
        gt_rotations:   list of (3, 3) GT 旋转矩阵
        sim3_params:    align_poses_sim3 返回的参数

    Returns:
        aligned_rotations: list of (3, 3) 对齐后的旋转矩阵
    """
    R_align = sim3_params['R']
    aligned = []
    for R_pred in pred_rotations:
        # 对齐后的旋转: R_world_to_cam_aligned = R_pred @ R_align^T
        # 因为 VGGT 坐标系经过 R_align 变到 GT 坐标系
        R_aligned = R_pred @ R_align.T
        aligned.append(R_aligned)
    return aligned


def rotation_error(R_pred, R_gt):
    """
    计算两个旋转矩阵之间的角度误差（度）。
    theta = arccos((tr(R_pred @ R_gt^T) - 1) / 2)
    """
    R_diff = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def relative_rotation_error(R_pred_i, R_pred_j, R_gt_i, R_gt_j):
    """
    计算相对旋转误差（不受全局坐标系影响）。
    R_rel_pred = R_pred_j @ R_pred_i^T
    R_rel_gt   = R_gt_j   @ R_gt_i^T
    """
    R_rel_pred = R_pred_j @ R_pred_i.T
    R_rel_gt = R_gt_j @ R_gt_i.T
    return rotation_error(R_rel_pred, R_rel_gt)


def relative_translation_angle_error(t_pred_i, t_pred_j, t_gt_i, t_gt_j):
    """
    计算相对平移方向的角度误差（度）。
    不受绝对尺度影响。
    """
    d_pred = t_pred_j - t_pred_i
    d_gt = t_gt_j - t_gt_i

    norm_pred = np.linalg.norm(d_pred)
    norm_gt = np.linalg.norm(d_gt)

    if norm_pred < 1e-10 or norm_gt < 1e-10:
        return float('nan')

    cos_angle = np.clip(np.dot(d_pred, d_gt) / (norm_pred * norm_gt), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))
