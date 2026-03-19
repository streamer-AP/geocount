"""
阶段一 Step 4 (Mock 版): 模拟 VGGT 推理

不需要 GPU，不需要下载模型，不需要数据集。
生成随机的相机参数，用于验证 Step 5/6 的评估和可视化流程是否正确。

用法:
    python scripts/step4_mock_vggt.py --dataset wildtrack
    python scripts/step4_mock_vggt.py --dataset multiviewx
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.coord_transform import extrinsic_to_camera_center


# Wildtrack GT 相机参数（硬编码的近似值，用于生成真实感的 mock 数据）
# 相机分布在约 [-5, 5] x [-5, 5] 的范围内，高度约 5-8m
WILDTRACK_APPROX_CENTERS = np.array([
    [-4.2,  3.1,  6.3],
    [-1.8,  4.5,  5.8],
    [ 1.2,  4.8,  6.1],
    [ 4.0,  3.3,  5.9],
    [ 4.5, -0.5,  6.4],
    [ 1.5, -4.2,  5.7],
    [-3.8, -3.5,  6.0],
])

MULTIVIEWX_APPROX_CENTERS = np.array([
    [-6.0,  4.0,  7.0],
    [-2.0,  6.0,  7.0],
    [ 2.0,  6.0,  7.0],
    [ 6.0,  4.0,  7.0],
    [ 6.0, -2.0,  7.0],
    [-6.0, -2.0,  7.0],
])


def make_rotation_look_at(camera_pos, target=np.array([0., 0., 0.]),
                          up=np.array([0., 0., 1.])):
    """
    构造 look-at 旋转矩阵：相机从 camera_pos 朝向 target。
    返回 R（3x3），满足 x_cam = R @ x_world + t 的 R 部分。
    """
    forward = camera_pos - target
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([0., 1., 0.])
        right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)

    up_corrected = np.cross(forward, right)
    up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)

    # R 的行是相机坐标系的 XYZ 基向量（在世界坐标系中的表达）
    R = np.stack([right, up_corrected, forward], axis=0)  # (3, 3)
    return R


def generate_mock_cameras(dataset, noise_level=0.1):
    """
    生成 mock 相机参数。

    GT 相机参数：基于近似的真实相机位置
    VGGT 估计值：在 GT 基础上加高斯噪声（模拟估计误差）

    Parameters:
        dataset: 'wildtrack' or 'multiviewx'
        noise_level: 噪声标准差（米）

    Returns:
        gt_cameras, pred_cameras: 各为 dict {cam_id -> {intrinsic, R, t, center}}
    """
    if dataset == 'wildtrack':
        centers = WILDTRACK_APPROX_CENTERS
        # Wildtrack 相机内参（近似）: 1920x1080, 焦距约 1100px
        fx, fy, cx, cy = 1100.0, 1100.0, 960.0, 540.0
    else:
        centers = MULTIVIEWX_APPROX_CENTERS
        # MultiviewX 相机内参（近似）: 1920x1080, 焦距约 1250px
        fx, fy, cx, cy = 1250.0, 1250.0, 960.0, 540.0

    K_gt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    gt_cameras = {}
    pred_cameras = {}
    n_cams = len(centers)

    rng = np.random.RandomState(42)

    for i, center in enumerate(centers):
        # --- GT 相机 ---
        R_gt = make_rotation_look_at(center)
        t_gt = -R_gt @ center

        gt_cameras[i] = {
            'intrinsic': K_gt.copy(),
            'R': R_gt,
            't': t_gt,
            'center': center.copy(),
        }

        # --- VGGT 估计（加噪声模拟估计误差）---
        # 位置噪声
        center_noisy = center + rng.randn(3) * noise_level
        center_noisy[2] = max(center_noisy[2], 1.0)  # 高度不能为负

        # 旋转噪声（对角度加小扰动）
        angle_noise = rng.randn(3) * np.radians(noise_level * 10)  # 约 1 度
        dR = _small_rotation(angle_noise)
        R_pred = dR @ make_rotation_look_at(center_noisy)
        t_pred = -R_pred @ center_noisy

        # 内参噪声（焦距误差约 5%）
        K_pred = K_gt.copy()
        K_pred[0, 0] *= (1 + rng.randn() * 0.05)
        K_pred[1, 1] *= (1 + rng.randn() * 0.05)
        K_pred[0, 2] += rng.randn() * 20  # 主点偏移约 20px
        K_pred[1, 2] += rng.randn() * 20

        # VGGT 输出格式: extrinsic (3, 4)
        E_pred = np.hstack([R_pred, t_pred.reshape(3, 1)])

        pred_cameras[i] = {
            'intrinsic': K_pred,
            'R': R_pred,
            't': t_pred,
            'center': extrinsic_to_camera_center(R_pred, t_pred),
            'extrinsic': E_pred,
        }

    # VGGT 输出是相对坐标系（以第一个相机为原点），需要模拟这个行为
    # 将所有 pred 位置转换到以第一个相机为参考的坐标系
    R0 = pred_cameras[0]['R']
    t0 = pred_cameras[0]['t']
    center0 = pred_cameras[0]['center']

    pred_cameras_vggt_frame = {}
    for i in range(n_cams):
        # 将相机中心从世界系转换到 cam0 系
        center_in_cam0 = R0 @ pred_cameras[i]['center'] + t0
        R_in_cam0 = pred_cameras[i]['R'] @ R0.T
        t_in_cam0 = -R_in_cam0 @ center_in_cam0
        E_in_cam0 = np.hstack([R_in_cam0, t_in_cam0.reshape(3, 1)])

        pred_cameras_vggt_frame[i] = {
            'intrinsic': pred_cameras[i]['intrinsic'],
            'R': R_in_cam0,
            't': t_in_cam0,
            'center': extrinsic_to_camera_center(R_in_cam0, t_in_cam0),
            'extrinsic': E_in_cam0,
        }

    return gt_cameras, pred_cameras_vggt_frame


def _small_rotation(angles_xyz):
    """构造小角度旋转矩阵（Rodriguez 公式近似）"""
    ax, ay, az = angles_xyz
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def save_mock_results(pred_cameras, output_dir, dataset, frame_id=0):
    """将 mock VGGT 输出保存为与真实推理相同的格式"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cams = len(pred_cameras)
    extrinsics = np.stack([pred_cameras[i]['extrinsic'] for i in range(n_cams)])  # (N, 3, 4)
    intrinsics = np.stack([pred_cameras[i]['intrinsic'] for i in range(n_cams)])  # (N, 3, 3)

    npz_path = output_dir / f"vggt_{dataset}_frame{frame_id}.npz"
    np.savez(str(npz_path),
             extrinsics=extrinsics,
             intrinsics=intrinsics,
             inference_time=np.array(0.01),
             num_views=np.array(n_cams))
    print(f"保存 mock VGGT 结果: {npz_path}")
    return npz_path


def save_mock_gt(gt_cameras, output_dir, dataset):
    """将 mock GT 标定保存为与真实解析相同的格式"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np_data = {}
    for cam_id, cam in gt_cameras.items():
        np_data[f"cam{cam_id}_intrinsic"] = cam['intrinsic']
        np_data[f"cam{cam_id}_extrinsic"] = np.hstack([cam['R'], cam['t'].reshape(3, 1)])
        np_data[f"cam{cam_id}_R"] = cam['R']
        np_data[f"cam{cam_id}_t"] = cam['t']
        np_data[f"cam{cam_id}_center"] = cam['center']

    npz_path = output_dir / "gt_cameras.npz"
    np.savez(str(npz_path), **np_data)
    print(f"保存 mock GT 标定: {npz_path}")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="Mock VGGT 推理（本地无 GPU 验证用）")
    parser.add_argument("--dataset", type=str, default="wildtrack",
                        choices=["wildtrack", "multiviewx"])
    parser.add_argument("--noise", type=float, default=0.1,
                        help="位置噪声标准差（米），模拟 VGGT 估计误差")
    parser.add_argument("--frame_id", type=int, default=0)
    args = parser.parse_args()

    print(f"=== Mock VGGT 推理 ===")
    print(f"数据集: {args.dataset}")
    print(f"噪声级别: {args.noise}m（旋转噪声: {args.noise * 10:.1f}度）")
    print("=" * 40)

    # 生成 mock 数据
    gt_cameras, pred_cameras = generate_mock_cameras(args.dataset, noise_level=args.noise)

    print(f"\n生成 {len(gt_cameras)} 个相机的 mock 数据")

    # 保存 mock GT（覆盖真实 GT，仅用于测试）
    gt_output_dir = ROOT / "results" / "gt_calibrations" / args.dataset
    save_mock_gt(gt_cameras, gt_output_dir, args.dataset)

    # 保存 mock VGGT 预测
    pred_output_dir = ROOT / "results" / "vggt_predictions"
    save_mock_results(pred_cameras, pred_output_dir, args.dataset, args.frame_id)

    # 打印摘要
    print("\n相机参数对比（GT vs Mock VGGT）:")
    print(f"  {'相机':>4} | {'GT 中心':>30} | {'VGGT 中心（相机0系）':>35}")
    print("  " + "-" * 75)
    for i in sorted(gt_cameras.keys()):
        c_gt = gt_cameras[i]['center']
        c_pred = pred_cameras[i]['center']
        print(f"  {i:>4} | ({c_gt[0]:6.2f}, {c_gt[1]:6.2f}, {c_gt[2]:6.2f}) | "
              f"({c_pred[0]:6.2f}, {c_pred[1]:6.2f}, {c_pred[2]:6.2f})")

    print("\n[完成] 现在可以运行后续步骤:")
    print(f"  python scripts/step5_evaluate.py --dataset {args.dataset}")
    print(f"  python scripts/step6_visualize.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
