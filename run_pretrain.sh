#!/bin/bash
#
# Cortex 预训练一键启动脚本
# 自动适配 CPU / GPU (CUDA) / NPU (Ascend)
#
# 用法:
#   ./run_pretrain.sh              # 单卡，自动检测设备
#   NPU=0,1 ./run_pretrain.sh      # NPU 多卡 (2卡)
#   NPU=0,1,2,3 ./run_pretrain.sh  # NPU 多卡 (4卡)
#   NPROC=4 ./run_pretrain.sh      # 4卡并行 + 自动检测可用卡
#
clear
set -e
cd "$(dirname "$0")"

# ---- 0. 环境初始化 ----
source /root/miniforge3/etc/profile.d/conda.sh
conda activate xcy

if [ -f /home/x30063410/env/ascend-toolkit/set_env.sh ]; then
    source /home/x30063410/env/ascend-toolkit/set_env.sh
fi

# ---- 1. 卡数配置 ----
# NPU: 指定用哪几张卡，默认自动检测全部
# NPROC: 并行进程数，默认 1（单卡）
if [ -n "$NPU" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="$NPU"
    NPROC=${NPROC:-$(echo "$NPU" | tr ',' '\n' | wc -l)}
else
    NPROC=${NPROC:-1}
fi

# ---- 2. 设备检测 ----
echo "========================================"
echo "  Cortex Pretraining Launcher"
echo "========================================"

echo -n "[检测] NPU 后端... "
if python3 -c "import torch_npu; print('ok')" 2>/dev/null; then
    echo "正常"
else
    echo "驱动缺失，已禁用 NPU 后端自动加载"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
fi

echo -n "[检测] 可用设备... "
DEVICE=$(
    python3 -c "
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '${TORCH_DEVICE_BACKEND_AUTOLOAD:-1}')
import torch
try:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        print(f'NPU Ascend ({torch.npu.device_count()}卡可用)')
    elif torch.cuda.is_available():
        print(f'GPU CUDA ({torch.cuda.device_count()}卡)')
    else:
        print('CPU')
except Exception:
    print('CPU')
" 2>/dev/null
)
echo "$DEVICE"

# ---- 3. 清理 ----
rm -f log/*.lock 2>/dev/null || true

# ---- 4. 启动 ----
echo ""
echo "启动时间: $(date)"
echo "并行模式: ${NPROC} 卡"
if [ "$NPROC" -gt 1 ]; then
    echo "可见设备: ${ASCEND_RT_VISIBLE_DEVICES:-auto}"
fi
echo "日志:     log/log.txt"
echo "模型:     ckpt_dir/model.pth"
echo "----------------------------------------"
echo "按 Ctrl+C 中断"
echo "========================================"
echo ""

export PYTHONUNBUFFERED=1

if [ "$NPROC" -gt 1 ]; then
    # ---- 多卡分布式 ----
    export PARALLEL_TYPE=ds
    torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
        train_pretrain.py 2>&1 | tee train_output.log
else
    # ---- 单卡 ----
    python3 -u train_pretrain.py 2>&1 | tee train_output.log
fi
