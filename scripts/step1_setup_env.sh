#!/bin/bash
# 阶段一 Step 1: 环境搭建 (Apple Silicon / 本地验证版)
# 用法: bash scripts/step1_setup_env.sh
#
# 环境: Apple M4, Python 3.11 (pyenv), 无 conda
# 加速: PyTorch MPS (Metal Performance Shaders)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "=== 创建虚拟环境 (Python 3.11) ==="
pyenv local 3.11.14
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo ""
echo "=== 安装 PyTorch (Apple MPS) ==="
# Apple Silicon 直接用默认 pip 版本，内置 MPS 支持
pip install torch torchvision

echo ""
echo "=== 安装 VGGT ==="
pip install vggt

echo ""
echo "=== 安装其他依赖 ==="
pip install opencv-python scipy matplotlib numpy tqdm

echo ""
echo "=== 验证安装 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built:     {torch.backends.mps.is_built()}')

from vggt.models.vggt import VGGT
print('VGGT import OK')
print()
print('环境搭建完成!')
print(f'激活虚拟环境: source {VENV_DIR}/bin/activate')
"

echo ""
echo ">>> 后续每次使用前激活环境:"
echo "    source $VENV_DIR/bin/activate"
