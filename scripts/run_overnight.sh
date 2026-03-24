#!/bin/bash
# 今晚 GPU 批处理任务
# 任务：MultiviewX 全部 10 帧推理 + 评估 + 可视化 + 多帧汇总
#
# 用法: bash scripts/run_overnight.sh > logs/overnight.log 2>&1 &

set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# 学术网络加速 + HuggingFace 镜像（模型已缓存则无影响）
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs

DATASET=multiviewx
TOTAL_FRAMES=10

echo "======================================="
echo "GeoCount 批处理任务开始"
echo "数据集: $DATASET"
echo "总帧数: $TOTAL_FRAMES"
echo "开始时间: $(date)"
echo "======================================="

# -------------------------------------------------------
# Step 3: 解析 GT 标定（只需一次）
# -------------------------------------------------------
echo
echo "[Step 3] 解析 GT 标定..."
python scripts/step3_parse_gt_calib.py --dataset $DATASET
echo "[Step 3] 完成"

# -------------------------------------------------------
# Step 4: VGGT 推理（跳过已有结果）
# -------------------------------------------------------
echo
echo "[Step 4] VGGT 推理（共 $TOTAL_FRAMES 帧）..."
for FRAME_ID in $(seq 0 $((TOTAL_FRAMES - 1))); do
    NPZ_PATH="results/vggt_predictions/vggt_${DATASET}_frame${FRAME_ID}.npz"
    if [ -f "$NPZ_PATH" ]; then
        echo "  帧 $FRAME_ID: 已存在，跳过"
    else
        echo "  帧 $FRAME_ID: 开始推理..."
        python scripts/step4_run_vggt.py \
            --dataset $DATASET \
            --frame_id $FRAME_ID \
            --batch_size 16
        echo "  帧 $FRAME_ID: 推理完成"
    fi
done
echo "[Step 4] 全部完成"

# -------------------------------------------------------
# Step 5: 评估（所有帧）
# -------------------------------------------------------
echo
echo "[Step 5] 评估相机参数精度..."
for FRAME_ID in $(seq 0 $((TOTAL_FRAMES - 1))); do
    echo "  帧 $FRAME_ID: 评估..."
    python scripts/step5_evaluate.py \
        --dataset $DATASET \
        --frame_id $FRAME_ID
    echo "  帧 $FRAME_ID: 完成"
done
echo "[Step 5] 全部完成"

# -------------------------------------------------------
# Step 6: 可视化（所有帧）
# -------------------------------------------------------
echo
echo "[Step 6] 生成可视化图表..."
for FRAME_ID in $(seq 0 $((TOTAL_FRAMES - 1))); do
    echo "  帧 $FRAME_ID: 可视化..."
    python scripts/step6_visualize.py \
        --dataset $DATASET \
        --frame_id $FRAME_ID 2>/dev/null || echo "  帧 $FRAME_ID: 可视化失败（非致命）"
done
echo "[Step 6] 全部完成"

# -------------------------------------------------------
# 多帧汇总分析
# -------------------------------------------------------
echo
echo "[汇总] 多帧误差统计..."
python scripts/analyze_multi_frame.py --dataset $DATASET

echo
echo "======================================="
echo "所有任务完成！"
echo "结束时间: $(date)"
echo "结果目录:"
echo "  推理: results/vggt_predictions/"
echo "  评估: results/evaluation/"
echo "  图表: results/figures/$DATASET/"
echo "  汇总: results/evaluation/multi_frame_summary_${DATASET}.json"
# repaired truncated tail
