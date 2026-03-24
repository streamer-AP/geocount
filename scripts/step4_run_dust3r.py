"""
Step 4 (替代): 运行 DUSt3R/MASt3R 推理

从 Wildtrack / MultiviewX 数据集中取同步多视角图像，
输入 DUSt3R 或 MASt3R 获取估计的相机参数。

输出格式与 step4_run_vggt.py 一致 (NPZ)，可直接用 step5_evaluate.py 评估。

用法:
    python scripts/step4_run_dust3r.py --dataset multiviewx --frame_id 0
    python scripts/step4_run_dust3r.py --dataset multiviewx --frame_id 0 --model mast3r
    python scripts/step4_run_dust3r.py --dataset wildtrack --frame_id 0 --niter 500
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# DUSt3R / MASt3R paths
MAST3R_ROOT = Path("/root/mast3r")
DUST3R_ROOT = MAST3R_ROOT / "dust3r"
sys.path.insert(0, str(MAST3R_ROOT))
sys.path.insert(0, str(DUST3R_ROOT))


def get_image_paths(data_dir, dataset, frame_id=0):
    """获取数据集中某一帧的所有视角图像路径。"""
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


def run_dust3r(image_paths, model_name="dust3r", niter=300, lr=0.01):
    """
    运行 DUSt3R 或 MASt3R 推理，获取相机参数。

    Parameters:
        image_paths: list of str, 图像路径
        model_name: "dust3r" 或 "mast3r"
        niter: 全局对齐迭代次数
        lr: 学习率

    Returns:
        dict: 包含 extrinsics, intrinsics 等，格式兼容 step5
    """
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")

    # 加载模型
    print(f"加载 {model_name.upper()} 模型...")
    t0 = time.time()

    if model_name == "mast3r":
        from mast3r.model import AsymmetricMASt3R
        model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        ).to(device)
    else:
        from dust3r.model import AsymmetricCroCo3DStereo
        model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(device)

    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    # 加载和预处理图像
    print("预处理图像...")
    images = load_images(image_paths, size=512)
    print(f"加载 {len(images)} 张图像")

    # 记录 DUSt3R 内部分辨率
    # load_images 返回 list of dict, 每个 dict 有 'true_shape' 和 'img' (3, H, W)
    resized_h, resized_w = images[0]['true_shape'][0]
    print(f"DUSt3R 内部分辨率: {resized_h}x{resized_w}")

    # 创建图像对 (complete graph for small set)
    n_views = len(image_paths)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f"图像对数量: {len(pairs)} (complete graph, {n_views} views)")

    # 运行 pairwise 推理
    print("运行 pairwise 推理...")
    t0 = time.time()
    output = inference(pairs, model, device, batch_size=1)
    inference_time = time.time() - t0
    print(f"Pairwise 推理完成 ({inference_time:.2f}s)")

    # 全局对齐
    print(f"运行全局对齐 (niter={niter}, lr={lr})...")
    t0 = time.time()

    if n_views == 2:
        mode = GlobalAlignerMode.PairViewer
    else:
        mode = GlobalAlignerMode.PointCloudOptimizer

    scene = global_aligner(output, device=device, mode=mode)
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule='cosine', lr=lr
    )
    align_time = time.time() - t0
    print(f"全局对齐完成 ({align_time:.2f}s, final loss={loss:.4f})")

    # 提取输出
    poses = scene.get_im_poses().detach().cpu().numpy()       # (N, 4, 4) cam-to-world
    intrinsics = scene.get_intrinsics().detach().cpu().numpy() # (N, 3, 3)

    # 转换 cam-to-world (4x4) → world-to-cam [R|t] (3x4)
    N = poses.shape[0]
    extrinsics = np.zeros((N, 3, 4), dtype=np.float64)
    for i in range(N):
        w2c = np.linalg.inv(poses[i])
        extrinsics[i] = w2c[:3, :]

    result = {
        'extrinsics': extrinsics,                          # (N, 3, 4)
        'intrinsics': intrinsics,                          # (N, 3, 3)
        'resized_hw': np.array([resized_h, resized_w]),    # DUSt3R 内部分辨率
        'inference_time': inference_time + align_time,
        'num_views': N,
    }

    # 提取 depth 和 point maps
    try:
        pts3d = scene.get_pts3d()
        if pts3d is not None:
            world_points = np.stack([p.detach().cpu().numpy() for p in pts3d])
            result['world_points'] = world_points
    except Exception as e:
        print(f"[警告] 无法获取 point maps: {e}")

    try:
        depthmaps = scene.get_depthmaps()
        if depthmaps is not None:
            depth = np.stack([d.detach().cpu().numpy() for d in depthmaps])
            result['depth'] = depth
    except Exception as e:
        print(f"[警告] 无法获取 depth maps: {e}")

    return result


def save_results(result, output_dir, dataset_name, frame_id, model_name):
    """保存推理结果为 NPZ（兼容 step5_evaluate.py）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            save_data[key] = val
        else:
            save_data[key] = np.array(val)

    # 使用与 VGGT 相同的命名格式，step5 可直接读取
    npz_path = output_dir / f"vggt_{dataset_name}_frame{frame_id}.npz"
    np.savez(str(npz_path), **save_data)
    print(f"\n保存结果: {npz_path}")

    # 打印摘要
    print("\n" + "=" * 50)
    print(f"{model_name.upper()} 估计的相机参数:")
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
    parser = argparse.ArgumentParser(description="运行 DUSt3R/MASt3R 推理")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["wildtrack", "multiviewx"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--frame_id", type=int, default=0,
                        help="使用哪一帧图像")
    parser.add_argument("--model", type=str, default="dust3r",
                        choices=["dust3r", "mast3r"],
                        help="使用 DUSt3R 还是 MASt3R")
    parser.add_argument("--niter", type=int, default=300,
                        help="全局对齐迭代次数")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="全局对齐学习率")
    args = parser.parse_args()

    if args.data_dir is None:
        dataset_name = "Wildtrack" if args.dataset == "wildtrack" else "MultiviewX"
        args.data_dir = ROOT / "data" / dataset_name

    print(f"数据集: {args.dataset}")
    print(f"路径: {args.data_dir}")
    print(f"帧: {args.frame_id}")
    print(f"模型: {args.model}")
    print(f"对齐迭代: {args.niter}")
    print("=" * 50)

    # 获取图像路径
    image_paths = get_image_paths(args.data_dir, args.dataset, args.frame_id)

    if not image_paths:
        print("[错误] 未找到图像！")
        return

    # 运行推理
    result = run_dust3r(
        image_paths,
        model_name=args.model,
        niter=args.niter,
        lr=args.lr,
    )

    # 保存到 vggt_predictions/ 目录（兼容现有 step5）
    output_dir = ROOT / "results" / "vggt_predictions"
    save_results(result, output_dir, args.dataset, args.frame_id, args.model)


if __name__ == "__main__":
    main()
