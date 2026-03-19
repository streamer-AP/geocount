"""
阶段一 Step 4: 运行 VGGT 推理

从 Wildtrack / MultiviewX 数据集中取同步多视角图像，
输入 VGGT 获取估计的相机参数。

用法:
    python scripts/step4_run_vggt.py --dataset wildtrack --frame_id 0
    python scripts/step4_run_vggt.py --dataset wildtrack --frame_id 0 --with_ba
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_image_paths_wildtrack(data_dir, frame_id=0):
    """
    获取 Wildtrack 数据集中某一帧的所有视角图像路径。
    Wildtrack 图像命名: C1/00000000.png, C2/00000000.png, ...
    """
    data_dir = Path(data_dir)
    img_dir = data_dir / "Image_subsets"

    cam_dirs = sorted(img_dir.glob("C*"))
    if not cam_dirs:
        raise FileNotFoundError(f"未找到相机目录: {img_dir}")

    paths = []
    for cam_dir in cam_dirs:
        frame_files = sorted(cam_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(cam_dir.glob("*.jpg"))

        if frame_id < len(frame_files):
            paths.append(str(frame_files[frame_id]))
        else:
            print(f"[警告] {cam_dir.name} 只有 {len(frame_files)} 帧，跳过")

    print(f"加载 {len(paths)} 个视角的图像（帧 {frame_id}）:")
    for p in paths:
        print(f"  {p}")

    return paths


def get_image_paths_multiviewx(data_dir, frame_id=0):
    """获取 MultiviewX 数据集中某一帧的所有视角图像路径。"""
    data_dir = Path(data_dir)
    img_dir = data_dir / "Image_subsets"

    cam_dirs = sorted(img_dir.glob("C*"))
    if not cam_dirs:
        raise FileNotFoundError(f"未找到相机目录: {img_dir}")

    paths = []
    for cam_dir in cam_dirs:
        frame_files = sorted(cam_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(cam_dir.glob("*.jpg"))

        if frame_id < len(frame_files):
            paths.append(str(frame_files[frame_id]))

    print(f"加载 {len(paths)} 个视角的图像（帧 {frame_id}）:")
    for p in paths:
        print(f"  {p}")

    return paths


def run_vggt(image_paths, with_ba=False):
    """
    运行 VGGT 推理，获取相机参数和其他 3D 输出。

    Parameters:
        image_paths: list of str, 图像路径
        with_ba: bool, 是否使用 Bundle Adjustment 后处理

    Returns:
        dict: 包含 extrinsics, intrinsics, depth_maps, world_points 等
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # 设备选择: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"\n设备: {device}, 精度: {dtype}")

    # 加载模型
    print("加载 VGGT 模型...")
    print("(首次运行需从 HuggingFace 下载约 4.5GB，请耐心等待)")
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).to(dtype)
    model.set_mode("all")
    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    # 加载和预处理图像
    print("预处理图像...")
    images = load_and_preprocess_images(image_paths)
    print(f"图像 tensor 形状: {images.shape}")

    # 推理
    print("运行推理...")
    t0 = time.time()
    with torch.no_grad():
        # MPS 和 CPU 不支持 autocast，只在 CUDA 下使用
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images.to(device))
        else:
            predictions = model(images.to(device))

    inference_time = time.time() - t0
    print(f"推理完成 ({inference_time:.2f}s)")

    # 提取相机参数
    pose_enc = predictions.get("pose_enc", None)

    if pose_enc is not None:
        image_shape = images.shape[-2:]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_shape
        )
        extrinsics = extrinsics.cpu().numpy()
        intrinsics = intrinsics.cpu().numpy()
    else:
        extrinsics = predictions["extrinsic"].cpu().numpy()
        intrinsics = predictions["intrinsic"].cpu().numpy()

    # 处理 batch 维度
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]  # (N, 3, 4)
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]  # (N, 3, 3)

    result = {
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        'inference_time': inference_time,
        'num_views': len(image_paths),
    }

    # 可选输出
    for key in ['depth', 'world_points', 'world_points_conf']:
        if key in predictions:
            val = predictions[key].cpu().numpy()
            if val.ndim > 3:
                val = val[0]
            result[key] = val

    # Bundle Adjustment 后处理
    if with_ba:
        print("\n运行 Bundle Adjustment...")
        try:
            from vggt.utils.bundle_adjustment import run_ba
            ba_result = run_ba(predictions, images)
            if 'extrinsic' in ba_result:
                result['extrinsics_ba'] = ba_result['extrinsic'].cpu().numpy()
                if result['extrinsics_ba'].ndim == 4:
                    result['extrinsics_ba'] = result['extrinsics_ba'][0]
                print("BA 完成")
        except (ImportError, RuntimeError) as e:
            print(f"BA 失败: {e}")

    return result


def save_vggt_results(result, output_dir, dataset_name, frame_id):
    """保存 VGGT 推理结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            save_data[key] = val
        else:
            save_data[key] = np.array(val)

    npz_path = output_dir / f"vggt_{dataset_name}_frame{frame_id}.npz"
    np.savez(str(npz_path), **save_data)
    print(f"\n保存结果: {npz_path}")

    # 打印相机参数摘要
    print("\n" + "=" * 50)
    print("VGGT 估计的相机参数:")
    print("=" * 50)

    for i in range(result['extrinsics'].shape[0]):
        E = result['extrinsics'][i]
        K = result['intrinsics'][i]
        R = E[:3, :3]
        t = E[:3, 3]
        center = -R.T @ t

        print(f"\n相机 {i}:")
        print(f"  焦距: ({K[0,0]:.1f}, {K[1,1]:.1f})")
        print(f"  主点: ({K[0,2]:.1f}, {K[1,2]:.1f})")
        print(f"  中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="运行 VGGT 推理")
    parser.add_argument("--dataset", type=str, default="wildtrack",
                        choices=["wildtrack", "multiviewx"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--frame_id", type=int, default=0,
                        help="使用哪一帧图像")
    parser.add_argument("--with_ba", action="store_true",
                        help="是否使用 Bundle Adjustment 后处理")
    args = parser.parse_args()

    if args.data_dir is None:
        dataset_name = "Wildtrack" if args.dataset == "wildtrack" else "MultiviewX"
        args.data_dir = ROOT / "data" / dataset_name

    print(f"数据集: {args.dataset}")
    print(f"路径: {args.data_dir}")
    print(f"帧: {args.frame_id}")
    print(f"BA: {args.with_ba}")
    print("=" * 50)

    # 获取图像路径
    if args.dataset == "wildtrack":
        image_paths = get_image_paths_wildtrack(args.data_dir, args.frame_id)
    else:
        image_paths = get_image_paths_multiviewx(args.data_dir, args.frame_id)

    if not image_paths:
        print("[错误] 未找到图像！")
        return

    # 运行 VGGT
    result = run_vggt(image_paths, with_ba=args.with_ba)

    # 保存
    output_dir = ROOT / "results" / "vggt_predictions"
    save_vggt_results(result, output_dir, args.dataset, args.frame_id)


if __name__ == "__main__":
    main()
