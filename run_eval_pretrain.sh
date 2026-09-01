#!/bin/bash
#
# Cortex 预训练测评一键脚本（常识 / 补全准确率）
#
# 用法:
#   ./run_eval_pretrain.sh
#   ./run_eval_pretrain.sh --ckpt ./last_checkpoint.bin
#   ./run_eval_pretrain.sh --bench-file ./eval_data/pretrain_bench.jsonl
#
set -e
cd "$(dirname "$0")"

CKPT=${CKPT:-./last_checkpoint.bin}

if [ ! -f "$CKPT" ]; then
    echo "[eval] 未找到 $CKPT，尝试从 ckpt_dir 导出..."
    if [ ! -f ./ckpt_dir/zero_to_fp32.py ]; then
        echo "[eval] 缺少 ckpt_dir/zero_to_fp32.py，请先完成预训练并保留 checkpoint"
        exit 1
    fi
    (
        cd ./ckpt_dir
        python3 zero_to_fp32.py ./ ../
    )
    if [ -f ./pytorch_model.bin ]; then
        mv ./pytorch_model.bin "$CKPT"
    else
        echo "[eval] 导出失败：未生成 pytorch_model.bin"
        exit 1
    fi
fi

echo "[eval] ckpt=$CKPT"
python3 -u eval_pretrain.py --ckpt "$CKPT" "$@"
