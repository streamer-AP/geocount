#!/bin/bash
# ============================================================
# 今晚 GPU 批处理实验 (2026-03-21)
#
# 实验清单:
#   Phase 1B: Depth map 验证 (3 种模式 × 10 帧, CPU 为主, 快)
#   Phase 2:  多帧 VGGT 推理 (pairs/triples/5-frame windows, GPU 密集)
#
# 用法: bash scripts/run_tonight.sh > logs/tonight.log 2>&1 &
#   或: nohup bash scripts/run_tonight.sh > logs/tonight.log 2>&1 &
#
# 监控: tail -f logs/tonight.log
# ============================================================

set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# 学术网络加速 + HuggingFace 镜像
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs

DATASET=multiviewx
START_TIME=$(date)

echo "========================================================"
echo "GeoCount 今晚实验批处理"
echo "日期: $START_TIME"
echo "数据集: $DATASET"
echo "========================================================"

# GPU 信息
if command -v nvidia-smi &>/dev/null; then
    echo
    echo "[GPU Info]"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

# ============================================================
# 实验 1: Phase 1B — Depth Map 验证 (快, 不需要 GPU 推理)
# ============================================================
echo
echo "########################################################"
echo "# 实验 1: Depth Map 验证 (Phase 1B)"
echo "#   Mode A: depth + GT_K + GT_Rt (upper bound)"
echo "#   Mode B: depth + GT_K + VGGT_Rt (部分免标定)"
echo "#   Mode C: depth + VGGT_K + VGGT_Rt (完全免标定)"
echo "########################################################"

echo
echo "[Step 8] Depth map validation (all modes, all frames)..."
python scripts/step8_validate_depth.py \
    --dataset $DATASET \
    --all_frames \
    --mode all
echo "[Step 8] 完成"

# ============================================================
# 实验 2: 多帧 VGGT 推理 — Frame Pairs (GPU)
# ============================================================
echo
echo "########################################################"
echo "# 实验 2: 多帧 VGGT — Frame Pairs (12 views each)"
echo "########################################################"

echo
echo "[Step 9 - Pairs] 多帧推理 (frame pairs)..."
python scripts/step9_multi_frame_vggt.py \
    --dataset $DATASET \
    --exp pair
echo "[Step 9 - Pairs] 完成"

# ============================================================
# 实验 3: 多帧 VGGT 推理 — Frame Triples (GPU)
# ============================================================
echo
echo "########################################################"
echo "# 实验 3: 多帧 VGGT — Frame Triples (18 views each)"
echo "########################################################"

echo
echo "[Step 9 - Triples] 多帧推理 (frame triples)..."
python scripts/step9_multi_frame_vggt.py \
    --dataset $DATASET \
    --exp triple
echo "[Step 9 - Triples] 完成"

# ============================================================
# 实验 4: 多帧 VGGT 推理 — 5-Frame Windows (GPU, 可能 OOM)
# ============================================================
echo
echo "########################################################"
echo "# 实验 4: 多帧 VGGT — 5-Frame Windows (30 views, 试探)"
echo "########################################################"

echo
echo "[Step 9 - Window5] 多帧推理 (5-frame windows)..."
python scripts/step9_multi_frame_vggt.py \
    --dataset $DATASET \
    --exp window5 || echo "[Step 9 - Window5] 失败 (可能 OOM, 非致命)"
echo "[Step 9 - Window5] 完成"

# ============================================================
# 汇总
# ============================================================
echo
echo "########################################################"
echo "# 汇总报告"
echo "########################################################"

echo
echo "--- Depth validation results ---"
if [ -f "results/depth_validation/depth_comparison_${DATASET}.json" ]; then
    python -c "
import json
with open('results/depth_validation/depth_comparison_${DATASET}.json') as f:
    d = json.load(f)
print('Depth Map Validation Results:')
for mode in ['A', 'B', 'C']:
    if mode in d:
        m = d[mode]
        err = m['error_xy_mean']['across_frames_mean']
        status = 'PASS' if m['gate_pass_xy'] else 'FAIL'
        print(f'  Mode {mode} ({m[\"mode_desc\"]}): {err:.3f}m [{status}]')
print()
print('Baseline (world_points): 3.614m [FAIL]')
"
fi

echo
echo "--- Multi-frame VGGT results ---"
for exp in pair triple window5; do
    json_file="results/multi_frame_vggt/${exp}_summary_${DATASET}.json"
    if [ -f "$json_file" ]; then
        python -c "
import json, numpy as np
with open('$json_file') as f:
    d = json.load(f)
rot_errs = [e['rel_rotation_error_mean'] for e in d]
pos_errs = [e['position_error_mean'] for e in d]
focal_errs = [e['focal_error_mean'] for e in d]
print(f'  ${exp}: rot={np.mean(rot_errs):.2f}° pos={np.mean(pos_errs):.3f}m focal={np.mean(focal_errs):.1f}%')
"
    fi
done

echo
echo "Baseline single-frame: rot=8.53° pos=1.21m focal=5.9%"

END_TIME=$(date)
echo
echo "========================================================"
echo "所有实验完成！"
echo "开始: $START_TIME"
echo "结束: $END_TIME"
echo "========================================================"
echo
echo "结果目录:"
echo "  Depth 验证: results/depth_validation/"
echo "  多帧 VGGT:  results/multi_frame_vggt/"
echo "  日志:       logs/tonight.log"
