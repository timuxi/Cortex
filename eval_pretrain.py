#!/usr/bin/env python3
"""
Cortex 预训练测评：常识 / 补全准确率（似然排序）。

对每个选择题，计算「query + choice」中 choice 段的平均 NLL，取最低者作为预测。
适合未做 SFT 的预训练模型（不依赖指令格式）。

用法:
  python3 eval_pretrain.py --ckpt ./last_checkpoint.bin
  python3 eval_pretrain.py --ckpt ./last_checkpoint.bin --bench-file ./eval_data/pretrain_bench.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from llm_model import LlmModel
from llm_trainer import TrainerTools
from llm_trainer.loss import chunked_linear_cross_entropy

from utils import get_model_config, init_env


DEFAULT_BENCH = os.path.join(os.path.dirname(__file__), 'eval_data', 'pretrain_bench.jsonl')


def _pick_device() -> torch.device:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device('npu:0')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def _needs_space(query: str, choice: str) -> bool:
    """英文 query 与 choice 之间补空格；中文一般直接拼接。"""
    if not query or not choice:
        return False
    left = query[-1]
    right = choice[0]
    # 两侧都是 ASCII 字母/数字时加空格
    return left.isascii() and left.isalnum() and right.isascii() and right.isalnum()


def _join_query_choice(query: str, choice: str) -> str:
    if _needs_space(query, choice):
        return f'{query} {choice}'
    return f'{query}{choice}'


def load_bench(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ('id', 'query', 'choices', 'answer'):
                if key not in obj:
                    raise ValueError(f'{path}:{line_no} missing field `{key}`')
            if not isinstance(obj['choices'], list) or len(obj['choices']) < 2:
                raise ValueError(f'{path}:{line_no} choices must have >= 2 items')
            if not (0 <= int(obj['answer']) < len(obj['choices'])):
                raise ValueError(f'{path}:{line_no} invalid answer index')
            items.append(obj)
    return items


def _load_model(ckpt: str, device: torch.device) -> LlmModel:
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f'checkpoint not found: {ckpt}\n'
            '请先导出：cd ckpt_dir && python3 zero_to_fp32.py ./ ../ && '
            'cd .. && mv pytorch_model.bin last_checkpoint.bin'
        )

    config = get_model_config(long_context=False)
    model = LlmModel(config)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[eval] warning missing keys: {len(missing)} (show 5) {missing[:5]}')
    if unexpected:
        print(f'[eval] warning unexpected keys: {len(unexpected)} (show 5) {unexpected[:5]}')

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_completion_nll(
        model: LlmModel,
        context: str,
        completion: str,
        device: torch.device,
        chunk_size: int,
        max_seq_len: int,
) -> float:
    """
    返回 completion 段的平均 token NLL（越低越像模型会接的续写）。
    通过构造 labels：context 部分为 -100，只对 completion token 计 loss。
    """
    tok = TrainerTools().tokenizer
    ctx_ids = tok.encode(context, unsqueeze=False, covert_tensor=False)
    full_text = _join_query_choice(context, completion)
    full_ids = tok.encode(full_text, unsqueeze=False, covert_tensor=False)

    # 容错：若 encode(query+choice) 与 encode(query)+encode(choice) 不完全可切，
    # 用最长公共前缀对齐 context 长度
    prefix_len = 0
    for a, b in zip(ctx_ids, full_ids):
        if a != b:
            break
        prefix_len += 1
    if prefix_len == 0 and len(ctx_ids) > 0:
        # 退化为整句计分（仍可比较相对高低）
        prefix_len = 0

    if len(full_ids) < 2:
        return float('inf')

    # 截断到模型上下文
    if len(full_ids) > max_seq_len:
        overflow = len(full_ids) - max_seq_len
        full_ids = full_ids[overflow:]
        prefix_len = max(0, prefix_len - overflow)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    # 不计 context（以及 shift 后对应位置）；用 -100 忽略
    # LMLoss/chunked CE 会 labels[:, 1:]，因此把「不评分」位置标 -100
    labels[0, :prefix_len] = -100

    # 若 completion 被截光，无法评分
    if (labels[0, 1:] != -100).sum().item() == 0:
        return float('inf')

    out = model(input_ids, return_logits=False)
    loss = chunked_linear_cross_entropy(
        out['hidden_states'],
        model.lm_head.weight,
        labels,
        ignore_index=-100,
        chunk_size=chunk_size,
    )
    return float(loss.item())


@torch.no_grad()
def eval_multiple_choice(
        model: LlmModel,
        items: List[dict],
        device: torch.device,
        chunk_size: int,
) -> Tuple[dict, List[dict]]:
    max_seq_len = model.config.max_position_embeddings
    details: List[dict] = []
    correct = 0
    by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

    for i, item in enumerate(items):
        scores = []
        for choice in item['choices']:
            nll = score_completion_nll(
                model,
                item['query'],
                str(choice),
                device=device,
                chunk_size=chunk_size,
                max_seq_len=max_seq_len,
            )
            scores.append(nll)

        pred = int(min(range(len(scores)), key=lambda j: scores[j]))
        gold = int(item['answer'])
        ok = pred == gold
        if ok:
            correct += 1

        t = item.get('type', 'unknown')
        by_type[t]['total'] += 1
        if ok:
            by_type[t]['correct'] += 1

        details.append({
            'id': item['id'],
            'type': t,
            'query': item['query'],
            'choices': item['choices'],
            'answer': gold,
            'prediction': pred,
            'correct': ok,
            'scores_nll': scores,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(
                f'[eval] {i + 1}/{len(items)} '
                f'acc={correct / (i + 1):.4f}'
            )

    metrics: Dict[str, Any] = {
        'accuracy': correct / max(len(items), 1),
        'correct': correct,
        'total': len(items),
        'by_type': {
            k: {
                'accuracy': v['correct'] / max(v['total'], 1),
                'correct': v['correct'],
                'total': v['total'],
            }
            for k, v in sorted(by_type.items())
        },
    }
    return metrics, details


def main():
    parser = argparse.ArgumentParser(
        description='Cortex pretrain eval: common-sense / completion accuracy'
    )
    parser.add_argument('--ckpt', type=str, default='./last_checkpoint.bin')
    parser.add_argument(
        '--bench-file',
        type=str,
        default=DEFAULT_BENCH,
        help='jsonl：每行含 id/query/choices/answer[/type]',
    )
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--output-dir', type=str, default='./eval_pretrain_out')
    parser.add_argument(
        '--max-items',
        type=int,
        default=0,
        help='只评前 N 题，0 表示全部',
    )
    args = parser.parse_args()

    init_env()
    os.makedirs(args.output_dir, exist_ok=True)

    device = _pick_device()
    print(f'[eval] device={device}')
    print(f'[eval] bench={args.bench_file}')

    items = load_bench(args.bench_file)
    if args.max_items > 0:
        items = items[: args.max_items]
    print(f'[eval] num_items={len(items)}')

    model = _load_model(args.ckpt, device)
    metrics, details = eval_multiple_choice(
        model, items, device=device, chunk_size=args.chunk_size
    )
    metrics.update({
        'ckpt': os.path.abspath(args.ckpt),
        'bench_file': os.path.abspath(args.bench_file),
        'device': str(device),
        'method': 'completion_mean_nll_ranking',
    })

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    details_path = os.path.join(args.output_dir, 'details.jsonl')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(details_path, 'w', encoding='utf-8') as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(
        f'[eval] accuracy={metrics["accuracy"]:.4f} '
        f'({metrics["correct"]}/{metrics["total"]})'
    )
    for t, v in metrics['by_type'].items():
        print(f'  - {t}: {v["accuracy"]:.4f} ({v["correct"]}/{v["total"]})')
    print(f'[eval] wrote {metrics_path}')
    print(f'[eval] wrote {details_path}')


if __name__ == '__main__':
    main()
