#!/bin/bash
# 阶段一 Step 2: 数据下载
# 用法: bash scripts/step2_download_data.sh
#
# 注意: Wildtrack 需要手动下载（需同意协议）
# MultiviewX 可通过 git clone 获取

set -e

DATA_DIR="data"
mkdir -p $DATA_DIR

echo "============================================"
echo "  数据下载指南"
echo "============================================"

echo ""
echo "=== 1. Wildtrack 数据集 ==="
echo "需要手动下载:"
echo "  访问: https://www.epfl.ch/labs/cvlab/data/data-wildtrack/"
echo "  下载 Wildtrack 完整数据集，解压到 data/Wildtrack/"
echo ""
echo "预期目录结构:"
echo "  data/Wildtrack/"
echo "  ├── Image_subsets/"
echo "  │   ├── C1/ ... C7/     (七个视角的图像)"
echo "  ├── calibrations/"
echo "  │   ├── extrinsic/       (外参 XML)"
echo "  │   └── intrinsic_zero/  (去畸变内参 XML)"
echo "  └── annotations_positions/"

echo ""
echo "=== 2. MultiviewX 数据集 ==="
echo "通过 git clone 下载:"

if [ ! -d "$DATA_DIR/MultiviewX" ]; then
    echo "正在克隆 MultiviewX..."
    cd $DATA_DIR
    git clone https://github.com/hou-yz/MultiviewX.git
    cd ..
    echo "MultiviewX 下载完成"
else
    echo "MultiviewX 目录已存在，跳过下载"
fi

echo ""
echo "=== 验证数据 ==="
echo ""

# 验证 Wildtrack
if [ -d "$DATA_DIR/Wildtrack/Image_subsets" ]; then
    NUM_CAMS=$(ls -d $DATA_DIR/Wildtrack/Image_subsets/C* 2>/dev/null | wc -l)
    echo "[OK] Wildtrack: 找到 $NUM_CAMS 个相机视角"
else
    echo "[!!] Wildtrack: 未找到，请手动下载到 data/Wildtrack/"
fi

# 验证 MultiviewX
if [ -d "$DATA_DIR/MultiviewX" ]; then
    echo "[OK] MultiviewX: 已下载"
else
    echo "[!!] MultiviewX: 未找到"
fi

echo ""
echo "下载完成后，运行: python scripts/step3_parse_gt_calib.py"
