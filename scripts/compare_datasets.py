"""跨数据集对比分析

对比 MultiviewX（合成）与 Wildtrack（真实）上 VGGT 的相机参数估计精度，
生成并打印对比报告，保存到 results/evaluation/comparison_report.json。

用法:
    python scripts/compare_datasets.py
    python scripts/compare_datasets.py --datasets wildtrack multiviewx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 评估阈值（来自 CLAUDE.md）
THRESHOLDS = {
    "relative_rotation_error_deg":      [(2,  "优秀"), (5,  "可用"), (float("inf"), "需改进")],
    "position_error_m":                 [(0.2,"优秀"), (0.5,"可用"), (float("inf"), "需改进")],
    "reprojection_error_px":            [(10, "优秀"), (30, "可用"), (float("inf"), "需改进")],
    "focal_error_pct":                  [(5,  "优秀"), (15, "可用"), (float("inf"), "需改进")],
}

METRIC_LABELS = {
    "relative_rotation_error_deg":      "相对旋转误差 (°)",
    "relative_translation_angle_deg":   "相对平移角度误差 (°)",
    "position_error_m":                 "位置误差 (m)",
    "focal_error_pct":                  "焦距误差 (%)",
    "reprojection_error_px":            "重投影误差 (px)",
}

DATASET_LABELS = {
    "multiviewx": "MultiviewX（合成）",
    "wildtrack":  "Wildtrack（真实）",
}


def grade(val, metric_key):
    if metric_key not in THRESHOLDS or val is None or np.isnan(val):
        return "—"
    for thresh, label in THRESHOLDS[metric_key]:
        if val < thresh:
            return label
    return "需改进"


def load_summary(dataset):
    """加载多帧汇总 JSON"""
    path = ROOT / "results" / "evaluation" / f"multi_frame_summary_{dataset}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_per_frame(dataset):
    """加载所有逐帧评估 JSON，提取每帧每相机的指标"""
    eval_dir = ROOT / "results" / "evaluation"
    files = sorted(eval_dir.glob(f"evaluation_{dataset}_frame*.json"))
    frames = []
    for fpath in files:
        frame_id = int(fpath.stem.split("frame")[1])
        with open(fpath) as f:
            data = json.load(f)
        reproj = data.get("reprojection", {})
        reproj_means = [v["mean"] for v in reproj.values()
                        if isinstance(v.get("mean"), float) and not np.isnan(v["mean"])]
        frames.append({
            "frame_id": frame_id,
            "summary": data["summary"],
            "reproj_mean": float(np.mean(reproj_means)) if reproj_means else None,
            "sim3_scale": data["sim3_params"]["scale"],
        })
    return frames


def extract_metric(summary, key):
    """从 summary 中提取均值，支持 reprojection_error_px"""
    if key == "reprojection_error_px":
        v = summary.get("reprojection_error_px", {})
        return v.get("mean") if v else None
    s = summary.get(key, {})
    return s.get("mean") if s else None


def print_comparison_table(results):
    """打印横向对比表"""
    datasets = list(results.keys())
    col_w = 28

    # 表头
    header = f"{'指标':<30}"
    for ds in datasets:
        label = DATASET_LABELS.get(ds, ds)
        header += f"  {label:>{col_w}}"
    print(header)
    print("-" * (30 + (col_w + 2) * len(datasets)))

    for key, label in METRIC_LABELS.items():
        row = f"{label:<30}"
        for ds in datasets:
            if ds not in results or results[ds] is None:
                row += f"  {'(无数据)':>{col_w}}"
                continue
            val = extract_metric(results[ds]["summary"], key)
            if val is None:
                row += f"  {'—':>{col_w}}"
            else:
                unit_val = f"{val:.3f}"
                g = grade(val, key)
                cell = f"{unit_val} [{g}]"
                row += f"  {cell:>{col_w}}"
        print(row)

    # Sim3 scale 行
    row = f"{'Sim3 尺度 (均值)':<30}"
    for ds in datasets:
        if ds not in results or results[ds] is None:
            row += f"  {'(无数据)':>{col_w}}"
            continue
        scale = results[ds].get("sim3_scale_mean")
        if scale is None:
            row += f"  {'—':>{col_w}}"
        else:
            row += f"  {scale:.4f}{' ':>{col_w - 6}}"
    print(row)

    # 帧数行
    row = f"{'评估帧数':<30}"
    for ds in datasets:
        if ds not in results or results[ds] is None:
            row += f"  {'(无数据)':>{col_w}}"
            continue
        n = results[ds].get("num_frames", "?")
        row += f"  {str(n):>{col_w}}"
    print(row)


def print_per_frame_detail(frames, dataset):
    """打印逐帧明细"""
    if not frames:
        print("  (无数据)")
        return

    label = DATASET_LABELS.get(dataset, dataset)
    print(f"\n{label} 逐帧结果:")
    print(f"  {'帧':>4} | {'位置(m)':>8} | {'旋转(°)':>8} | {'焦距(%)':>8} | {'重投影(px)':>10} | Sim3_scale")
    print("  " + "-" * 62)
    for r in frames:
        sm = r["summary"]
        pos  = sm.get("position_error_m", {}).get("mean", float("nan"))
        rot  = sm.get("relative_rotation_error_deg", {}).get("mean", float("nan"))
        foc  = sm.get("focal_error_pct", {}).get("mean", float("nan"))
        rep  = r["reproj_mean"] if r["reproj_mean"] is not None else float("nan")
        sc   = r["sim3_scale"]
        print(f"  {r['frame_id']:>4} | {pos:>8.3f} | {rot:>8.2f} | {foc:>8.2f} | {rep:>10.1f} | {sc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="跨数据集对比分析")
    parser.add_argument("--datasets", nargs="+",
                        default=["multiviewx", "wildtrack"],
                        choices=["multiviewx", "wildtrack"],
                        help="要对比的数据集列表")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GeoCount Stage-1 跨数据集对比报告")
    print("=" * 70)

    results = {}
    all_frames = {}

    for ds in args.datasets:
        summary = load_summary(ds)
        frames = load_per_frame(ds)

        if summary is None and not frames:
            print(f"\n[警告] {ds}: 未找到任何评估结果，跳过。")
            print(f"  运行以下命令生成结果:")
            if ds == "wildtrack":
                print(f"    bash scripts/run_wildtrack.sh")
            else:
                print(f"    bash scripts/run_overnight.sh")
            continue

        if summary is None and frames:
            # 从逐帧数据重建 summary
            print(f"\n[信息] {ds}: 未找到汇总文件，从逐帧数据重建...")
            metrics_agg = {k: [] for k in METRIC_LABELS if k != "reprojection_error_px"}
            reproj_list = []
            for r in frames:
                for k in metrics_agg:
                    v = r["summary"].get(k, {}).get("mean")
                    if v is not None:
                        metrics_agg[k].append(v)
                if r["reproj_mean"] is not None:
                    reproj_list.append(r["reproj_mean"])

            summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                       for k, v in metrics_agg.items() if v}
            if reproj_list:
                summary["reprojection_error_px"] = {
                    "mean": float(np.mean(reproj_list)),
                    "std": float(np.std(reproj_list)),
                }

        # Sim3 scale 均值
        sim3_scales = [r["sim3_scale"] for r in frames]
        # multi_frame_summary JSON 结构: {dataset, num_frames, per_frame, summary:{...}}
        inner_summary = summary.get("summary", summary)
        results[ds] = {
            "summary": inner_summary,
            "num_frames": summary.get("num_frames", len(frames)),
            "sim3_scale_mean": float(np.mean(sim3_scales)) if sim3_scales else None,
        }
        all_frames[ds] = frames

    if not results:
        print("\n没有任何数据集有评估结果，退出。")
        return

    # -------------------------------------------------------
    # 打印对比表
    # -------------------------------------------------------
    print("\n【核心指标对比】")
    print()
    print_comparison_table(results)

    # -------------------------------------------------------
    # 打印逐帧明细
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    for ds in args.datasets:
        if ds in all_frames:
            print_per_frame_detail(all_frames[ds], ds)

    # -------------------------------------------------------
    # 可行性综合结论
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("可行性综合结论")
    print("=" * 70)
    for ds, res in results.items():
        label = DATASET_LABELS.get(ds, ds)
        print(f"\n{label}:")
        sm = res["summary"]
        for key, metric_label in METRIC_LABELS.items():
            val = extract_metric(sm, key)
            if val is None:
                continue
            g = grade(val, key)
            print(f"  {metric_label:<28} {val:>8.3f}  ->  {g}")

    # -------------------------------------------------------
    # 合成 vs 真实差距分析（如果两者都有）
    # -------------------------------------------------------
    if "multiviewx" in results and "wildtrack" in results:
        print("\n" + "=" * 70)
        print("合成 vs 真实 差距分析")
        print("=" * 70)
        mx = results["multiviewx"]["summary"]
        wt = results["wildtrack"]["summary"]
        for key, label in METRIC_LABELS.items():
            v_mx = extract_metric(mx, key)
            v_wt = extract_metric(wt, key)
            if v_mx is None or v_wt is None:
                continue
            delta = v_wt - v_mx
            sign = "+" if delta >= 0 else ""
            pct_change = (delta / v_mx * 100) if v_mx != 0 else float("nan")
            print(f"  {label:<28}  Δ = {sign}{delta:.3f}  ({sign}{pct_change:.1f}%)")

    # -------------------------------------------------------
    # 保存 JSON
    # -------------------------------------------------------
    output = {
        "datasets": {ds: {
            "num_frames": res["num_frames"],
            "sim3_scale_mean": res["sim3_scale_mean"],
            "summary": res["summary"],
        } for ds, res in results.items()},
    }

    out_path = ROOT / "results" / "evaluation" / "comparison_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n对比报告已保存: {out_path}")


if __name__ == "__main__":
    main()
