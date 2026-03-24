"""多帧误差汇总分析脚本

对所有已完成评估的帧进行统计，生成多帧平均误差报告。

用法:
    python scripts/analyze_multi_frame.py --dataset multiviewx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="多帧误差汇总分析")
    parser.add_argument("--dataset", type=str, default="multiviewx",
                        choices=["wildtrack", "multiviewx"])
    args = parser.parse_args()

    eval_dir = ROOT / "results" / "evaluation"
    result_files = sorted(eval_dir.glob(f"evaluation_{args.dataset}_frame*.json"))

    if not result_files:
        print(f"[错误] 未找到评估结果：{eval_dir}/evaluation_{args.dataset}_frame*.json")
        return

    print(f"找到 {len(result_files)} 个评估结果文件")
    print("=" * 70)

    all_metrics = {
        "position_error_m": [],
        "focal_error_pct": [],
        "relative_rotation_error_deg": [],
        "relative_translation_angle_deg": [],
    }
    all_reproj_mean = []
    frame_results = []

    for fpath in result_files:
        frame_id = int(fpath.stem.split("frame")[1])
        with open(fpath) as f:
            data = json.load(f)

        summary = data["summary"]
        reproj = data["reprojection"]
        reproj_means = [v["mean"] for v in reproj.values() if not (isinstance(v["mean"], float) and v["mean"] != v["mean"])]
        reproj_mean = float(np.mean(reproj_means)) if reproj_means else float("nan")

        row = {
            "frame_id": frame_id,
            "pos_err": summary["position_error_m"]["mean"],
            "focal_err": summary["focal_error_pct"]["mean"],
            "rot_err": summary["relative_rotation_error_deg"]["mean"],
            "trans_err": summary["relative_translation_angle_deg"]["mean"],
            "reproj_err": reproj_mean,
            "sim3_scale": data["sim3_params"]["scale"],
        }
        frame_results.append(row)

        for key in all_metrics:
            all_metrics[key].append(summary[key]["mean"])
        all_reproj_mean.append(reproj_mean)

    # 打印逐帧结果
    print(f"{'帧':>4} | {'位置误差(m)':>10} | {'焦距误差(%)':>10} | {'相对旋转(°)':>10} | {'相对平移(°)':>10} | {'重投影(px)':>10} | Sim3_scale")
    print("-" * 80)
    for r in frame_results:
        print(f"  {r['frame_id']:2d} | {r['pos_err']:10.3f} | {r['focal_err']:10.2f} | "
              f"{r['rot_err']:10.2f} | {r['trans_err']:10.2f} | "
              f"{r['reproj_err']:10.1f} | {r['sim3_scale']:.4f}")

    print()
    print("=" * 70)
    print("多帧统计汇总")
    print("=" * 70)

    metric_labels = {
        "position_error_m": "位置误差(m)",
        "focal_error_pct": "焦距误差(%)",
        "relative_rotation_error_deg": "相对旋转误差(°)",
        "relative_translation_angle_deg": "相对平移角度误差(°)",
    }
    for key, label in metric_labels.items():
        vals = all_metrics[key]
        print(f"  {label}:")
        print(f"    均值={np.mean(vals):.3f}  中位数={np.median(vals):.3f}  "
              f"标准差={np.std(vals):.3f}  最大={np.max(vals):.3f}  最小={np.min(vals):.3f}")

    valid_reproj = [v for v in all_reproj_mean if not np.isnan(v)]
    if valid_reproj:
        print(f"  重投影误差(px):")
        print(f"    均值={np.mean(valid_reproj):.1f}  中位数={np.median(valid_reproj):.1f}  "
              f"标准差={np.std(valid_reproj):.1f}  最大={np.max(valid_reproj):.1f}")

    print()
    print("=" * 70)
    print("可行性判断")
    print("=" * 70)

    def grade(val, thresholds, labels=("优秀", "可用", "需改进")):
        if val < thresholds[0]:
            return labels[0]
        elif val < thresholds[1]:
            return labels[1]
        return labels[2]

    rre = np.mean(all_metrics["relative_rotation_error_deg"])
    pos = np.mean(all_metrics["position_error_m"])
    focal = np.mean(all_metrics["focal_error_pct"])
    reproj = np.mean(valid_reproj) if valid_reproj else float("nan")

    print(f"  相对旋转误差:  {rre:.2f}°  -> {grade(rre, [2, 5])}")
    print(f"  位置误差:      {pos:.3f}m  -> {grade(pos, [0.2, 0.5])}")
    print(f"  焦距误差:      {focal:.2f}%  -> {grade(focal, [5, 15])}")
    print(f"  重投影误差:    {reproj:.1f}px -> {grade(reproj, [10, 30])}")

    # 保存汇总 JSON
    output = {
        "dataset": args.dataset,
        "num_frames": len(frame_results),
        "per_frame": frame_results,
        "summary": {
            "position_error_m": {"mean": float(np.mean(all_metrics["position_error_m"])),
                                  "std": float(np.std(all_metrics["position_error_m"]))},
            "focal_error_pct": {"mean": float(np.mean(all_metrics["focal_error_pct"])),
                                 "std": float(np.std(all_metrics["focal_error_pct"]))},
            "relative_rotation_error_deg": {"mean": float(np.mean(all_metrics["relative_rotation_error_deg"])),
                                             "std": float(np.std(all_metrics["relative_rotation_error_deg"]))},
            "relative_translation_angle_deg": {"mean": float(np.mean(all_metrics["relative_translation_angle_deg"])),
                                                "std": float(np.std(all_metrics["relative_translation_angle_deg"]))},
            "reprojection_error_px": {"mean": float(np.mean(valid_reproj)) if valid_reproj else None,
                                       "std": float(np.std(valid_reproj)) if valid_reproj else None},
        }
    }

    out_path = ROOT / "results" / "evaluation" / f"multi_frame_summary_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {out_path}")


if __name__ == "__main__":
    main()
