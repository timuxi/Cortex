import os
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


def _reset_peak_memory():
    """重置当前设备的峰值显存统计（不含历史峰值）。"""
    import torch
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _read_peak_memory_gib():
    """返回 (allocated_gib, reserved_gib, device_tag)。"""
    import torch
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return (
            torch.npu.max_memory_allocated() / 1024 ** 3,
            torch.npu.max_memory_reserved() / 1024 ** 3,
            f'npu:{torch.npu.current_device()}',
        )
    if torch.cuda.is_available():
        return (
            torch.cuda.max_memory_allocated() / 1024 ** 3,
            torch.cuda.max_memory_reserved() / 1024 ** 3,
            f'cuda:{torch.cuda.current_device()}',
        )
    return 0.0, 0.0, 'cpu'


class _MfuTimer:
    """记录每个 optimizer step 耗时。"""

    def __init__(self):
        self.step_times = []
        self._last = None

    def __call__(self):
        now = time.perf_counter()
        if self._last is not None:
            self.step_times.append(now - self._last)
        self._last = now


class _ProfileStepHook:
    """optimizer step 回调：MFU 计时 + torch_npu.profiler.step()。"""

    def __init__(self, timer: _MfuTimer, profiler=None):
        self.timer = timer
        self.profiler = profiler

    def __call__(self):
        self.timer()
        if self.profiler is not None:
            self.profiler.step()


def _make_npu_profiler(prof_dir: str, skip_first: int, active: int):
    """模型 ready 后再创建；schedule 跳过前 skip_first 步，只录 active 步。"""
    from torch_npu.profiler import (
        profile,
        schedule,
        ProfilerActivity,
        tensorboard_trace_handler,
        _ExperimentalConfig,
        ProfilerLevel,
        AiCMetrics,
    )

    os.makedirs(prof_dir, exist_ok=True)
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule(
            wait=0,
            warmup=0,
            active=active,
            repeat=1,
            skip_first=skip_first,
        ),
        on_trace_ready=tensorboard_trace_handler(prof_dir),
        experimental_config=_ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization,
        ),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cortex pretrain / 整网性能采集')
    parser.add_argument('--profile-steps', type=int, default=0,
                        help='性能采集总步数（>0 开启；默认 10：前 8 步 skip，采第 9、10 步；torch_npu.profiler）')
    parser.add_argument('--prof-dir', type=str, default='./prof',
                        help='torch_npu.profiler 输出目录（仅 --profile-steps>0 时生效）')
    parser.add_argument('--prof-skip-first', type=int, default=8,
                        help='profiler schedule：跳过前 N 个 optimizer step（不含冷启动）')
    parser.add_argument('--prof-active', type=int, default=2,
                        help='profiler schedule：实际采集的 step 数')
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
        # 强制 grad_accum=1 + 限步数；步数至少覆盖 skip+active
        need_steps = args.prof_skip_first + args.prof_active
        if profile_steps < need_steps:
            profile_steps = need_steps
        apply_profile_config(train_config, profile_steps)

    # 冷启动（建模 / DeepSpeed）在此完成，profiler 尚未开启
    trainer = Trainer(train_config=train_config, eval_prompts=eval_prompts)

    if profile_steps:
        num_params = _count_params(trainer.train_model)
        tokens_per_step = train_config.batch_size * train_config.dataset_block_size
        peak_tflops = args.peak_flops if args.peak_flops > 0 else _detect_peak_flops_tflops()

        # 模型已加载后再清峰值，统计训练阶段（含已驻留的权重）最高水位
        _reset_peak_memory()

        timer = _MfuTimer()
        rank = os.environ.get('RANK', '0')
        print(f'[PROF] rank={rank} dir={args.prof_dir} '
              f'skip_first={args.prof_skip_first} active={args.prof_active} '
              f'total_steps={profile_steps}（不含 Trainer 初始化冷启动）')

        with _make_npu_profiler(args.prof_dir, args.prof_skip_first, args.prof_active) as prof:
            trainer.on_step = _ProfileStepHook(timer, prof)
            trainer.train()

        alloc_gib, reserved_gib, device_tag = _read_peak_memory_gib()
        print(f'[MEM] rank={rank} device={device_tag} '
              f'peak_allocated={alloc_gib:.2f} GiB peak_reserved={reserved_gib:.2f} GiB')

        if timer.step_times:
            # 仅用最后 active 个 step 间隔估算 MFU
            n_active = max(1, args.prof_active)
            active_times = timer.step_times[-n_active:] if len(timer.step_times) >= n_active else timer.step_times
            avg_step_s = sum(active_times) / len(active_times)
            achieved_tflops = 6 * num_params * tokens_per_step / (avg_step_s * 1e12) if avg_step_s > 0 else 0.0
            mfu = _compute_mfu(num_params, tokens_per_step, avg_step_s, peak_tflops)
            print(f'[MFU] params={num_params / 1e6:.1f}M tokens/step={tokens_per_step} '
                  f'measured_steps={len(active_times)} avg_step={avg_step_s * 1000:.1f}ms '
                  f'achieved={achieved_tflops:.2f} TFLOPS peak={peak_tflops:.0f} TFLOPS MFU={mfu * 100:.2f}%')
    else:
        trainer.train()
