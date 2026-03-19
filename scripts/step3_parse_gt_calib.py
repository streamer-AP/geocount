"""
阶段一 Step 3: 解析 GT 标定参数

将 Wildtrack / MultiviewX 数据集的 XML 标定文件解析为统一的 numpy 格式，
并保存到 results/gt_calibrations/ 目录。

用法:
    python scripts/step3_parse_gt_calib.py --dataset wildtrack
    python scripts/step3_parse_gt_calib.py --dataset multiviewx
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.coord_transform import extrinsic_to_camera_center


# ============================================================
#  XML 解析工具
# ============================================================

def parse_opencv_xml(xml_path):
    """
    解析 OpenCV 风格的 XML 文件，提取所有矩阵。
    返回 dict: tag_name -> numpy array
    """
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开: {xml_path}")

    result = {}
    # 获取根节点下所有 key
    root = fs.root()
    for key in root.keys():
        node = fs.getNode(key)
        if not node.empty():
            mat = node.mat()
            if mat is not None:
                result[key] = mat
    fs.release()
    return result


def parse_wildtrack_calibration(data_dir):
    """
    解析 Wildtrack 数据集的标定参数。

    Wildtrack 目录结构:
        calibrations/
        ├── extrinsic/
        │   ├── extr_CVLab1.xml  (包含 rvec 和 tvec)
        │   └── ...
        └── intrinsic_zero/
            ├── intr_CVLab1.xml  (包含 camera_matrix)
            └── ...
    """
    data_dir = Path(data_dir)
    calib_dir = data_dir / "calibrations"

    if not calib_dir.exists():
        raise FileNotFoundError(f"标定目录不存在: {calib_dir}")

    cameras = {}

    # 查找所有外参文件来确定相机数量
    extr_dir = calib_dir / "extrinsic"
    intr_dir = calib_dir / "intrinsic_zero"

    extr_files = sorted(extr_dir.glob("extr_CVLab*.xml"))
    if not extr_files:
        # 尝试其他命名模式
        extr_files = sorted(extr_dir.glob("*.xml"))

    print(f"找到 {len(extr_files)} 个相机标定文件")

    for cam_idx, extr_file in enumerate(extr_files):
        cam_name = extr_file.stem  # e.g., "extr_CVLab1"
        cam_num = cam_name.replace("extr_CVLab", "").replace("extr_", "")

        # --- 外参 ---
        extr_data = parse_opencv_xml(extr_file)

        # Wildtrack 外参文件通常包含 rvec 和 tvec
        rvec = None
        tvec = None
        for key, val in extr_data.items():
            key_lower = key.lower()
            if 'rvec' in key_lower or 'rotation' in key_lower:
                rvec = val.flatten()
            elif 'tvec' in key_lower or 'translation' in key_lower:
                tvec = val.flatten()

        if rvec is None or tvec is None:
            # 可能是直接存储旋转矩阵的格式
            print(f"  [警告] {extr_file.name}: 未找到 rvec/tvec，尝试其他格式")
            print(f"  可用的 key: {list(extr_data.keys())}")
            for key, val in extr_data.items():
                print(f"    {key}: shape={val.shape}")
            continue

        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        tvec = tvec.reshape(3)

        # --- 内参 ---
        # 查找对应的内参文件
        intr_file = intr_dir / f"intr_CVLab{cam_num}.xml"
        if not intr_file.exists():
            intr_file = intr_dir / extr_file.name.replace("extr_", "intr_")
        if not intr_file.exists():
            intr_files_found = sorted(intr_dir.glob("*.xml"))
            if cam_idx < len(intr_files_found):
                intr_file = intr_files_found[cam_idx]

        K = np.eye(3)
        if intr_file.exists():
            intr_data = parse_opencv_xml(intr_file)
            for key, val in intr_data.items():
                if val.shape == (3, 3):
                    K = val
                    break
        else:
            print(f"  [警告] 未找到内参文件: {intr_file}")

        # 计算相机中心
        center = extrinsic_to_camera_center(R, tvec)

        cameras[cam_idx] = {
            'name': f"C{cam_num}",
            'intrinsic': K,
            'extrinsic': np.hstack([R, tvec.reshape(3, 1)]),
            'R': R,
            't': tvec,
            'rvec': rvec,
            'center': center,
        }

        print(f"  相机 {cam_num}: "
              f"焦距=({K[0,0]:.1f}, {K[1,1]:.1f}), "
              f"中心=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

    return cameras


def parse_multiviewx_calibration(data_dir):
    """
    解析 MultiviewX 数据集的标定参数。
    格式与 Wildtrack 类似。
    """
    data_dir = Path(data_dir)
    calib_dir = data_dir / "calibrations"

    if not calib_dir.exists():
        raise FileNotFoundError(f"标定目录不存在: {calib_dir}")

    cameras = {}

    extr_dir = calib_dir / "extrinsic"
    intr_dir = calib_dir / "intrinsic"

    # MultiviewX 可能使用不同的命名
    extr_files = sorted(extr_dir.glob("*.xml"))
    if not extr_files:
        # 尝试在 calibrations 根目录查找
        extr_files = sorted(calib_dir.glob("extr*.xml"))

    print(f"找到 {len(extr_files)} 个相机标定文件")

    for cam_idx, extr_file in enumerate(extr_files):
        extr_data = parse_opencv_xml(extr_file)

        rvec = None
        tvec = None
        for key, val in extr_data.items():
            key_lower = key.lower()
            if 'rvec' in key_lower or 'rotation' in key_lower:
                rvec = val.flatten()
            elif 'tvec' in key_lower or 'translation' in key_lower:
                tvec = val.flatten()

        if rvec is None or tvec is None:
            print(f"  [警告] {extr_file.name}: keys={list(extr_data.keys())}")
            continue

        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        tvec = tvec.reshape(3)

        # 内参
        K = np.eye(3)
        intr_files = sorted(intr_dir.glob("*.xml")) if intr_dir.exists() else []
        if cam_idx < len(intr_files):
            intr_data = parse_opencv_xml(intr_files[cam_idx])
            for key, val in intr_data.items():
                if val.shape == (3, 3):
                    K = val
                    break

        center = extrinsic_to_camera_center(R, tvec)

        cameras[cam_idx] = {
            'name': f"C{cam_idx + 1}",
            'intrinsic': K,
            'extrinsic': np.hstack([R, tvec.reshape(3, 1)]),
            'R': R,
            't': tvec,
            'rvec': rvec,
            'center': center,
        }

        print(f"  相机 {cam_idx + 1}: "
              f"焦距=({K[0,0]:.1f}, {K[1,1]:.1f}), "
              f"中心=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

    return cameras


# ============================================================
#  保存
# ============================================================

def save_cameras(cameras, output_dir):
    """将解析后的相机参数保存为 .npz 和可读的 .json"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为 npz（方便加载）
    np_data = {}
    for cam_id, cam in cameras.items():
        np_data[f"cam{cam_id}_intrinsic"] = cam['intrinsic']
        np_data[f"cam{cam_id}_extrinsic"] = cam['extrinsic']
        np_data[f"cam{cam_id}_R"] = cam['R']
        np_data[f"cam{cam_id}_t"] = cam['t']
        np_data[f"cam{cam_id}_center"] = cam['center']

    npz_path = output_dir / "gt_cameras.npz"
    np.savez(str(npz_path), **np_data)
    print(f"\n保存 npz: {npz_path}")

    # 保存可读的 JSON 摘要
    summary = {}
    for cam_id, cam in cameras.items():
        summary[str(cam_id)] = {
            'name': cam['name'],
            'focal_length': [float(cam['intrinsic'][0, 0]),
                             float(cam['intrinsic'][1, 1])],
            'principal_point': [float(cam['intrinsic'][0, 2]),
                                float(cam['intrinsic'][1, 2])],
            'center': cam['center'].tolist(),
        }

    json_path = output_dir / "gt_cameras_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"保存 JSON: {json_path}")


def load_cameras(npz_path):
    """从 npz 文件加载相机参数"""
    data = np.load(str(npz_path))
    cameras = {}

    # 提取相机 ID
    cam_ids = set()
    for key in data.files:
        parts = key.split('_')
        cam_id = int(parts[0].replace('cam', ''))
        cam_ids.add(cam_id)

    for cam_id in sorted(cam_ids):
        cameras[cam_id] = {
            'intrinsic': data[f"cam{cam_id}_intrinsic"],
            'extrinsic': data[f"cam{cam_id}_extrinsic"],
            'R': data[f"cam{cam_id}_R"],
            't': data[f"cam{cam_id}_t"],
            'center': data[f"cam{cam_id}_center"],
        }

    return cameras


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="解析 GT 标定参数")
    parser.add_argument("--dataset", type=str, default="wildtrack",
                        choices=["wildtrack", "multiviewx"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据集路径（默认: data/<Dataset>）")
    args = parser.parse_args()

    if args.data_dir is None:
        dataset_name = "Wildtrack" if args.dataset == "wildtrack" else "MultiviewX"
        args.data_dir = ROOT / "data" / dataset_name

    print(f"数据集: {args.dataset}")
    print(f"路径: {args.data_dir}")
    print("=" * 50)

    if args.dataset == "wildtrack":
        cameras = parse_wildtrack_calibration(args.data_dir)
    else:
        cameras = parse_multiviewx_calibration(args.data_dir)

    if not cameras:
        print("\n[错误] 未解析到任何相机参数！请检查数据集路径和文件格式。")
        return

    # 保存
    output_dir = ROOT / "results" / "gt_calibrations" / args.dataset
    save_cameras(cameras, output_dir)

    # 打印摘要
    print("\n" + "=" * 50)
    print(f"共解析 {len(cameras)} 个相机")

    centers = np.array([cam['center'] for cam in cameras.values()])
    print(f"相机分布范围:")
    print(f"  X: [{centers[:, 0].min():.2f}, {centers[:, 0].max():.2f}]")
    print(f"  Y: [{centers[:, 1].min():.2f}, {centers[:, 1].max():.2f}]")
    print(f"  Z: [{centers[:, 2].min():.2f}, {centers[:, 2].max():.2f}]")


if __name__ == "__main__":
    main()
