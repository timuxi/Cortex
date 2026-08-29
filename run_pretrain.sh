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
#   NPU=0,1,2,3 ./run_pretrain.sh --prof   # 整网性能采集：前 8 步 warmup，采第 9、10 步；结果在 ./prof
#
clear
set -e
cd "$(dirname "$0")"

# ---- 0. 解析参数 ----
ENABLE_PROF=0
for arg in "$@"; do
    case "$arg" in
        --prof) ENABLE_PROF=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "未知参数: $arg（仅支持 --prof）" >&2
            exit 1
            ;;
    esac
done

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
# 旧版 torch_npu.profiler 残留目录，避免干扰本次采集
rm -rf export_only_prof_dir result_dir 2>/dev/null || true
if [ "$ENABLE_PROF" -eq 1 ]; then
    rm -rf ./prof 2>/dev/null || true
fi

# ---- 4. 启动 ----
echo ""
echo "启动时间: $(date)"
echo "并行模式: ${NPROC} 卡"
if [ "$NPROC" -gt 1 ]; then
    echo "可见设备: ${ASCEND_RT_VISIBLE_DEVICES:-auto}"
fi
echo "日志:     log/log.txt"
echo "模型:     ckpt_dir/model.pth"
if [ "$ENABLE_PROF" -eq 1 ]; then
    echo "性能采集: 开启（前 8 步 warmup，采第 9、10 步）"
    echo "prof 输出: ./prof"
fi
echo "----------------------------------------"
echo "按 Ctrl+C 中断"
echo "========================================"
echo ""

export PYTHONUNBUFFERED=1

# 性能采集：共 10 个 optimizer step，前 8 步 warmup，第 9、10 步出结果
PROF_STEPS=10
PROF_DIR=./prof

if [ "$ENABLE_PROF" -eq 1 ]; then
    mkdir -p "$PROF_DIR"
    PEAK_FLOP_ARGS=""
    if [ -n "${PEAK_FLOPS:-}" ]; then
        PEAK_FLOP_ARGS="--peak-flops ${PEAK_FLOPS}"
    fi

    # 本机 msprof 要求 --output=DIR（等号），空格形式会报 expected one argument
    if [ "$NPROC" -gt 1 ]; then
        export PARALLEL_TYPE=ds
        msprof --output="$PROF_DIR" \
            torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
            train_pretrain.py --profile-steps "${PROF_STEPS}" ${PEAK_FLOP_ARGS} \
            2>&1 | tee train_output.log
    else
        msprof --output="$PROF_DIR" \
            python3 -u train_pretrain.py --profile-steps "${PROF_STEPS}" ${PEAK_FLOP_ARGS} \
            2>&1 | tee train_output.log
    fi
elif [ "$NPROC" -gt 1 ]; then
    export PARALLEL_TYPE=ds
    torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
        train_pretrain.py 2>&1 | tee train_output.log
else
    python3 -u train_pretrain.py 2>&1 | tee train_output.log
fi
