#!/bin/bash
# Wildtrack 真实数据集批处理实验
#
# 用法:
#   bash scripts/run_wildtrack.sh [num_frames] [start_frame] [batch_size]
#   示例:
#     bash scripts/run_wildtrack.sh          # 默认: 10 帧
#     bash scripts/run_wildtrack.sh 5        # 前 5 帧
#     bash scripts/run_wildtrack.sh 10 0 16  # 10 帧，batch_size=16
#
# 后台运行（推荐）:
#   bash scripts/run_wildtrack.sh > logs/wildtrack.log 2>&1 &
#   tail -f logs/wildtrack.log

set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# 学术网络加速 + HuggingFace 镜像（模型已缓存则无影响）
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs

# -------------------------------------------------------
# 参数解析
# -------------------------------------------------------
NUM_FRAMES=${1:-10}         # 要跑的帧数
START_FRAME=${2:-0}         # 起始帧 ID
BATCH_SIZE=${3:-4}          # VGGT batch_size
DATASET=wildtrack
DATA_DIR="$ROOT/data/Wildtrack"

echo "======================================="
echo "GeoCount 真实数据集实验"
echo "数据集:   $DATASET"
echo "数据路径: $DATA_DIR"
echo "帧范围:   $START_FRAME ~ $((START_FRAME + NUM_FRAMES - 1))"
echo "总帧数:   $NUM_FRAMES"
echo "batch_size: $BATCH_SIZE"
echo "开始时间: $(date)"
echo "======================================="

# -------------------------------------------------------
# 前置检查
# -------------------------------------------------------
echo
echo "[检查] 验证数据集是否存在..."

if [ ! -d "$DATA_DIR" ]; then
    echo "[错误] Wildtrack 数据集未找到: $DATA_DIR"
    echo ""
    echo "Wildtrack 为真实监控数据集，需手动下载："
    echo "  1. 访问 https://www.epfl.ch/labs/cvlab/data/data-wildtrack/"
    echo "  2. 申请访问权限并下载"
    echo "  3. 解压到 $DATA_DIR"
    echo ""
    echo "目录结构应为："
    echo "  $DATA_DIR/"
    echo "  ├── Image_subsets/"
    echo "  │   ├── C1/  (00000000.png ~ 00001999.png)"
    echo "  │   ├── C2/"
    echo "  │   └── ... (7 个相机)"
    echo "  └── calibrations/"
    echo "      ├── extrinsic/  (extr_CVLab1.xml ...)"
    echo "      └── intrinsic_zero/  (intr_CVLab1.xml ...)"
    exit 1
fi

if [ ! -d "$DATA_DIR/calibrations" ]; then
    echo "[错误] 未找到标定文件目录: $DATA_DIR/calibrations"
    exit 1
fi

if [ ! -d "$DATA_DIR/Image_subsets" ]; then
    echo "[错误] 未找到图像目录: $DATA_DIR/Image_subsets"
    exit 1
fi

# 统计可用图像帧数
AVAIL_FRAMES=$(ls "$DATA_DIR/Image_subsets/C1/" 2>/dev/null | wc -l)
echo "[检查] 相机 C1 可用帧数: $AVAIL_FRAMES"

END_FRAME=$((START_FRAME + NUM_FRAMES - 1))
if [ "$END_FRAME" -ge "$AVAIL_FRAMES" ]; then
    echo "[警告] 请求帧范围 [$START_FRAME, $END_FRAME] 超出可用帧数 $AVAIL_FRAMES"
    END_FRAME=$((AVAIL_FRAMES - 1))
    NUM_FRAMES=$((END_FRAME - START_FRAME + 1))
    echo "[调整] 实际运行帧数: $NUM_FRAMES (帧 $START_FRAME ~ $END_FRAME)"
fi

echo "[检查] 通过"

# -------------------------------------------------------
# Step 3: 解析 GT 标定（只需一次）
# -------------------------------------------------------
GT_NPZ="$ROOT/results/gt_calibrations/$DATASET/gt_cameras.npz"
echo
echo "[Step 3] 解析 GT 标定参数..."
if [ -f "$GT_NPZ" ]; then
    echo "  已存在，跳过: $GT_NPZ"
else
    python scripts/step3_parse_gt_calib.py --dataset $DATASET
    echo "[Step 3] 完成"
fi

# -------------------------------------------------------
# Step 4: VGGT 推理（跳过已有结果）
# -------------------------------------------------------
echo
echo "[Step 4] VGGT 推理（共 $NUM_FRAMES 帧，batch_size=$BATCH_SIZE）..."

SKIPPED=0
INFERRED=0
FAILED=0

for FRAME_ID in $(seq $START_FRAME $END_FRAME); do
    NPZ_PATH="$ROOT/results/vggt_predictions/vggt_${DATASET}_frame${FRAME_ID}.npz"
    if [ -f "$NPZ_PATH" ]; then
        echo "  帧 $FRAME_ID: 已存在，跳过"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  帧 $FRAME_ID: 开始推理... ($(date +%H:%M:%S))"
        if python scripts/step4_run_vggt.py \
            --dataset $DATASET \
            --frame_id $FRAME_ID \
            --batch_size $BATCH_SIZE; then
            INFERRED=$((INFERRED + 1))
            echo "  帧 $FRAME_ID: 推理完成"
        else
            FAILED=$((FAILED + 1))
            echo "  帧 $FRAME_ID: [失败] 继续下一帧"
        fi
    fi
done

echo "[Step 4] 完成 (新推理=$INFERRED, 跳过=$SKIPPED, 失败=$FAILED)"

# -------------------------------------------------------
# Step 5: 评估
# -------------------------------------------------------
echo
echo "[Step 5] 评估相机参数精度..."

for FRAME_ID in $(seq $START_FRAME $END_FRAME); do
    NPZ_PATH="$ROOT/results/vggt_predictions/vggt_${DATASET}_frame${FRAME_ID}.npz"
    if [ ! -f "$NPZ_PATH" ]; then
        echo "  帧 $FRAME_ID: 跳过（推理结果不存在）"
        continue
    fi
    echo "  帧 $FRAME_ID: 评估..."
    python scripts/step5_evaluate.py \
        --dataset $DATASET \
        --frame_id $FRAME_ID 2>&1 | tail -20
    echo "  帧 $FRAME_ID: 完成"
done

echo "[Step 5] 全部完成"

# -------------------------------------------------------
# Step 6: 可视化
# -------------------------------------------------------
echo
echo "[Step 6] 生成可视化图表..."
mkdir -p "$ROOT/results/figures/$DATASET"

for FRAME_ID in $(seq $START_FRAME $END_FRAME); do
    EVAL_JSON="$ROOT/results/evaluation/evaluation_${DATASET}_frame${FRAME_ID}.json"
    if [ ! -f "$EVAL_JSON" ]; then
        echo "  帧 $FRAME_ID: 跳过（评估结果不存在）"
        continue
    fi
    echo "  帧 $FRAME_ID: 可视化..."
    python scripts/step6_visualize.py \
        --dataset $DATASET \
        --frame_id $FRAME_ID 2>/dev/null \
        && echo "  帧 $FRAME_ID: 完成" \
        || echo "  帧 $FRAME_ID: 可视化失败（非致命）"
done

echo "[Step 6] 全部完成"

# -------------------------------------------------------
# 多帧汇总分析
# -------------------------------------------------------
echo
echo "[汇总] 多帧误差统计..."
python scripts/analyze_multi_frame.py --dataset $DATASET

# -------------------------------------------------------
# 跨数据集对比（如果 MultiviewX 结果也存在）
# -------------------------------------------------------
MULTIVIEWX_SUMMARY="$ROOT/results/evaluation/multi_frame_summary_multiviewx.json"
WILDTRACK_SUMMARY="$ROOT/results/evaluation/multi_frame_summary_wildtrack.json"

if [ -f "$MULTIVIEWX_SUMMARY" ] && [ -f "$WILDTRACK_SUMMARY" ]; then
    echo
    echo "[对比] 跨数据集对比分析..."
    python scripts/compare_datasets.py
fi

echo
echo "======================================="
echo "Wildtrack 实验完成！"
echo "结束时间: $(date)"
echo "结果目录:"
echo "  推理: results/vggt_predictions/vggt_${DATASET}_frame*.npz"
echo "  评估: results/evaluation/evaluation_${DATASET}_frame*.json"
echo "  汇总: results/evaluation/multi_frame_summary_${DATASET}.json"
echo "  图表: results/figures/$DATASET/"
echo "======================================="
