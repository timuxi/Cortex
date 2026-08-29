import time
import argparse

from llm_trainer import Trainer
from utils import init_env, get_pretrain_config, apply_profile_config


def _count_params(model) -> int:
    """统计模型总参数量（含 embedding）。DeepSpeed 引擎取其 .module。"""
    model = getattr(model, 'module', model)
    return sum(p.numel() for p in model.parameters())


def _detect_peak_flops_tflops() -> float:
    """按 NPU 型号返回 BF16/FP16 峰值算力（TFLOPS）。"""
    try:
        import torch_npu
        name = torch_npu.npu.get_device_name()
    except Exception:
        name = ''
    name_l = name.lower()
    if '910b2' in name_l:
        return 376.0
    elif '910b3' in name_l:
        return 295.0
    elif '910b4' in name_l:
        return 270
    
    return 270



def _compute_mfu(num_params: int, tokens: int, elapsed_s: float, peak_tflops: float) -> float:
    """MFU = 6ND / (peak_flops * t)，其中 6 = 前向 2 + 反向 4。"""
    if elapsed_s <= 0:
        return 0.0
    return 6 * num_params * tokens / (peak_tflops * 1e12 * elapsed_s)


class _MfuTimer:
    """记录每个 optimizer step 耗时（整网采集由外层 msprof 负责）。"""

    def __init__(self):
        self.step_times = []
        self._last = None

    def __call__(self):
        now = time.perf_counter()
        if self._last is not None:
            self.step_times.append(now - self._last)
        self._last = now


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cortex pretrain / 整网性能采集')
    parser.add_argument('--profile-steps', type=int, default=0,
                        help='性能采集总步数（>0 开启；默认 10：前 8 步 warmup，用第 9、10 步估 MFU；msprof 覆盖整段）')
    parser.add_argument('--peak-flops', type=float, default=0.0,
                        help='NPU 峰值算力（TFLOPS），0 表示按设备型号自动检测')
    args = parser.parse_args()

    init_env()

    eval_prompts = [
        '自来水直接冲洗生肉不仅冲不掉细菌',
    ]

    train_config = get_pretrain_config()
    profile_steps = args.profile_steps if args.profile_steps > 0 else 0

    if profile_steps:
        # 强制 grad_accum=1（1 步 = 1 个 64 batch）+ 限步数
        apply_profile_config(train_config, profile_steps)

    trainer = Trainer(train_config=train_config, eval_prompts=eval_prompts)

    if profile_steps:
        num_params = _count_params(trainer.train_model)
        tokens_per_step = train_config.batch_size * train_config.dataset_block_size
        peak_tflops = args.peak_flops if args.peak_flops > 0 else _detect_peak_flops_tflops()

        timer = _MfuTimer()
        trainer.on_step = timer
        trainer.train()

        if timer.step_times:
            # 仅用最后 2 个 step 间隔估算 MFU（对应第 9、10 步）
            active_times = timer.step_times[-2:] if len(timer.step_times) >= 2 else timer.step_times
            avg_step_s = sum(active_times) / len(active_times)
            achieved_tflops = 6 * num_params * tokens_per_step / (avg_step_s * 1e12) if avg_step_s > 0 else 0.0
            mfu = _compute_mfu(num_params, tokens_per_step, avg_step_s, peak_tflops)
            print(f'[MFU] params={num_params / 1e6:.1f}M tokens/step={tokens_per_step} '
                  f'measured_steps={len(active_times)} avg_step={avg_step_s * 1000:.1f}ms '
                  f'achieved={achieved_tflops:.2f} TFLOPS peak={peak_tflops:.0f} TFLOPS MFU={mfu * 100:.2f}%')
    else:
        trainer.train()
