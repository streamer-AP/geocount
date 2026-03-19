#!/bin/bash
# 阶段一: 一键运行全部步骤
# 用法: bash scripts/run_stage1.sh [dataset] [frame_id]
# 示例: bash scripts/run_stage1.sh wildtrack 0

set -e

DATASET=${1:-wildtrack}
FRAME_ID=${2:-0}

echo "============================================"
echo "  阶段一: VGGT 可行性验证"
echo "  数据集: $DATASET"
echo "  帧 ID:  $FRAME_ID"
echo "============================================"

echo ""
echo ">>> Step 3: 解析 GT 标定参数"
python scripts/step3_parse_gt_calib.py --dataset $DATASET

echo ""
echo ">>> Step 4: 运行 VGGT 推理"
python scripts/step4_run_vggt.py --dataset $DATASET --frame_id $FRAME_ID

echo ""
echo ">>> Step 5: 评估对比"
python scripts/step5_evaluate.py --dataset $DATASET --frame_id $FRAME_ID

echo ""
echo ">>> Step 6: 可视化"
python scripts/step6_visualize.py --dataset $DATASET --frame_id $FRAME_ID

echo ""
echo "============================================"
echo "  阶段一完成!"
echo "  结果目录: results/"
echo "  图表目录: results/figures/$DATASET/"
echo "============================================"
