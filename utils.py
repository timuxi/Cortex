import os
import random

import torch
from llm_trainer import train_configs
from llm_trainer.tools import compute_lr_scheduler_steps
from llm_model import ModelConfig, RoPEConfig, MoEConfig, AttnResConfig
from file_dataset import *
from constant import *

# 是否开启Attention Residuals
ENABLE_ATTN_RES = False

def init_env():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # NPU 缓存分配：减少碎片（若已设置则尊重用户环境变量）
    os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # HCCL：默认 200MB 偏小；过大又占显存（每通信域约 2*BUFFSIZE）。512MB 在带宽与显存间折中。
    os.environ.setdefault('HCCL_BUFFSIZE', '512')

    os.environ['TOKEN_DIR'] = './tokens/'
    os.environ['LOG_DIR'] = './log/'

    os.environ['CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    os.environ['SAVE_BEST_CHECKPOINT'] = '0'


def get_eval_prompt(content: str) -> str:
    is_think = random.random() > 0.5
    think_tag = '/think' if is_think else '/no think'
    chat_template = [
        {'role': 'system', 'content': f''},
        {'role': 'user', 'content': f'{content} {think_tag}'}
    ]

    chat_template = TrainerTools().tokenizer.apply_chat_template(chat_template, tokenizer=False)
    return f'{chat_template}<assistant>' + ('<think>' if is_think else '<think></think><answer>')


def get_model_config(long_context=False):
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=4096,
        intermediate_size=12288,

        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,

        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        use_qk_norm=True,
        attention_implementation='sdpa',

        moe_config=None,

        attn_res_config=AttnResConfig(
            num_blocks=2
        ) if ENABLE_ATTN_RES else None,

        rope_config=RoPEConfig(
            rope_type=rope_type,
            rope_theta=1e6,
        ),
    )


def calc_lr_schedular_args(
        train_stage: str,
        epochs: int,
        all_data_size: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        **kwargs
):
    if train_stage in ['pretrain', 'midtrain']:
        kwargs['max_warmup_iters'] = 2000

    warmup_iters, cosine_annealing_batches = compute_lr_scheduler_steps(
        train_stage=train_stage,
        epochs=epochs,
        all_data_size=all_data_size,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **kwargs
    )

    if TrainerTools().parallel.is_main_process:
        print(f'stage={train_stage}, warmup_iters={warmup_iters}, cosine_annealing_batches={cosine_annealing_batches}')

    return warmup_iters, cosine_annealing_batches


def _get_train_config(
        n_epochs: int,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig,
        train_stage: str
):
    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True) if os.path.exists('./last_checkpoint.bin') else None
    ref_checkpoint = torch.load('./sft.bin', weights_only=True) if os.path.exists('./sft.bin') else None

    if train_stage != 'pretrain':
        assert init_state_dict is not None

    if train_stage == 'ppo':
        assert ref_checkpoint is not None

    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero2Config(
            # HCCL_BUFFSIZE 已提到 512MB；bucket 保持 500MB，避免再堆大块临时缓冲导致 OOM
            reduce_bucket_size=5e8,
            allgather_bucket_size=5e8,
            overlap_comm=True,
            reduce_scatter=True,
            contiguous_gradients=True,
        ),
        activation_checkpointing=train_configs.DsActivationCheckpointingConfig(
            cpu_checkpointing=True
        ) if ENABLE_ATTN_RES else None
    )

    generate_config = train_configs.GenerateConfig(
        max_seq_len=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        # repetition_penalty=1.15,
        # exclude_penalty_tokens=TrainerTools().tokenizer.encode('\n'),
        suppress_tokens=None
    )

    data_loader_config = train_configs.DataLoaderConfig(
        pin_memory=True,
        num_workers=0,
        shuffle=False,
    )

    min_lr_ratio = 0.1
    pretrain_config = None
    sft_config = None
    ppo_config = None
    dpo_config = None
    grpo_config = None

    if train_stage == 'ppo':
        enable_lr_scheduler = True

        ppo_epochs = 2
        ppo_batch_size = 8
        # gradient_accumulation_steps = 10

        max_lr = 5e-7
        initial_lr = 1e-7
        min_lr_ratio = 1.0

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=10000,
            batch_size=real_batch_size,
            gradient_accumulation_steps=3,
            ppo_epochs=ppo_epochs,
            ppo_batch_size=ppo_batch_size
        )

        ppo_config = train_configs.PPOConfig(
            ppo_epochs=ppo_epochs,
            ppo_batch_size=ppo_batch_size,
            gradient_accumulation_steps=3,
            value_optim_config=train_configs.OptimConfig(
                enable_lr_scheduler=enable_lr_scheduler,
                initial_lr=5e-7,
                warmup_iters=warmup_iters,
                max_lr=2e-6,
                min_lr=2e-6,
                cosine_annealing_period=period
            ),
            vf_coef=0.5,
            kl_beta=0.05,
            kl_estimator='k3',
            normalize_rewards=True,
            normalize_method='RunningMeanStd',
            ref_model_checkpoint=ref_checkpoint,
            generate_config=generate_config,
        )
    elif train_stage == 'dpo':
        enable_lr_scheduler = True

        max_lr = 5e-6
        initial_lr = 5e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=13000, # 13098
            batch_size=real_batch_size,
            gradient_accumulation_steps=3
        )

        dpo_config = train_configs.DPOConfig(
            ref_model_checkpoint=ref_checkpoint,
            mask_prompt=True,
            gradient_accumulation_steps=3,
            loss_beta=0.2,
            loss_label_smoothing=0.0,
            nll_loss_coef=0.2
        )
    elif train_stage == 'grpo':
        enable_lr_scheduler = True

        grpo_epochs = 2
        grpo_batch_size = 5
        grpo_group_size = 3

        max_lr = 1e-5
        initial_lr = 1e-6
        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=100000,
            batch_size=real_batch_size,
            gradient_accumulation_steps=3,
            grpo_epochs=grpo_epochs,
            grpo_batch_size=grpo_batch_size,
            group_size=grpo_group_size
        )

        grpo_config = train_configs.GRPOConfig(
            grpo_epochs=grpo_epochs,
            grpo_batch_size=grpo_batch_size,
            group_size=grpo_group_size,
            gradient_accumulation_steps=3,
            loss_beta=0.0,
            loss_clip_eps=3e-4,
            loss_clip_eps_high=4e-4,
            loss_importance_sampling_level='sequence',
            generate_config=generate_config,
        )
    elif train_stage == 'sft':
        enable_lr_scheduler = True
        max_lr = 2e-5
        initial_lr = 1e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=211700,  # 211740
            batch_size=real_batch_size,
            gradient_accumulation_steps=3,
        )

        sft_config = train_configs.SFTConfig(
            mask_prompt=True,
            gradient_accumulation_steps=3,
            kd_config=None
        )
    elif train_stage == 'midtrain':
        enable_lr_scheduler = True
        max_lr = 6e-5
        initial_lr = 1e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=507300,  # 507374
            batch_size=real_batch_size,
            gradient_accumulation_steps=3,
        )

        pretrain_config = train_configs.PretrainConfig(
            gradient_accumulation_steps=3,
            kd_config=None
        )
    else:
        enable_lr_scheduler = True
        max_lr = 6e-4
        initial_lr = 1e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=6600100,  # 6600128
            batch_size=real_batch_size,
            gradient_accumulation_steps=3,
        )

        pretrain_config = train_configs.PretrainConfig(
            gradient_accumulation_steps=3,
            kd_config=None
        )

    optim_config = train_configs.OptimConfig(
        enable_lr_scheduler=enable_lr_scheduler,
        auto_optimize_optimizer=False,
        initial_lr=initial_lr,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=max_lr * min_lr_ratio,
        cosine_annealing_period=period
    )

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        dataset_block_size=model_config.max_position_embeddings,
        loss_config=train_configs.LossConfig(
            # 分块 linear+CE，避免物化 [B,S,V]；峰值约 chunk_size×V（2048×32k×2≈128MiB）
            chunk_size=2048,
        ),
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        init_state_dict=init_state_dict,
        save_and_eval_interval=10 if train_stage == 'grpo' or train_stage == 'ppo' else 100,
        eval_config=generate_config,
        pretrain_config=pretrain_config,
        sft_config=sft_config,
        ppo_config=ppo_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config
    )

    return train_config


def get_pretrain_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=1,
        file_dataset=PretrainFileDataset(),
        model_config=get_model_config(long_context=False),
        train_stage='pretrain'
    )


def get_midtrain_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=10,
        file_dataset=MidtrainFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='midtrain'
   )


def get_sft_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=10,
        file_dataset=SFTFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='sft'
    )


def get_ppo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=24,
        file_dataset=PPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='ppo'
    )


def get_dpo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=10,
        file_dataset=DPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='dpo'
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=3,
        file_dataset=GRPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='grpo'
    )


def apply_profile_config(train_config, profile_steps: int):
    """
    性能采集模式（配合 train_pretrain 内 torch_npu.profiler schedule）：
    - 强制 gradient_accumulation_steps=1，使 1 个 optimizer step 正好 = 1 个 batch 的前向+反向。
    - 限制最多执行 profile_steps 个 optimizer step（默认 10：前 8 步 skip，采第 9、10 步估 MFU）。
    - Profiler 在 Trainer/模型初始化完成之后才开启，不包含冷启动。
    """
    for cfg in (train_config.pretrain_config, train_config.sft_config, train_config.dpo_config):
        if cfg is not None:
            cfg.gradient_accumulation_steps = 1

    train_config.max_steps = profile_steps
    return train_config
