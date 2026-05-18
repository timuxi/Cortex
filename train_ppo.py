import os
import re
import concurrent.futures
from typing import List, Optional
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from openai import OpenAI

from llm_trainer import PPOTrainer, TrainerTools
from utils import init_env, get_ppo_config, get_eval_prompt

init_env()

# ==========================================
# LLM Judge 配置 (兼容 OpenAI SDK)
# ==========================================
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.siliconflow.cn/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V3.2")

assert JUDGE_API_KEY != ''
assert JUDGE_BASE_URL != ''
assert JUDGE_MODEL != ''

client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)


def call_llm_judge(prompt: str, answer: str, is_think_required: bool) -> float:
    """
    调用 LLM API 进行打分，极简 Prompt 防止裁判废话，包含安全拦截机制
    """

    safe_answer = answer.replace("<think>", "[思考开始]") \
        .replace("</think>", "[思考结束]") \
        .replace("<answer>", "[回答开始]") \
        .replace("</answer>", "[回答结束]")

    base_prompt = """你是一个冷酷客观的 AI 打分机器。你需要评估一个极小AI模型的输出格式与逻辑。
请【完全忽略】事实准确性！只看格式、排版和语言连贯性。
你可以先用一两句话简评，但【文章最后必须且只能】输出一个严格的 XML 标签：<score>得分</score>。范围是 -5.0 到 5.0。"""

    if is_think_required:
        rule_prompt = """该题【必须包含思考过程】。标准结构：[思考开始]思考内容[思考结束][回答开始]回答内容[回答结束]。
    [4.0 到 5.0]：[思考开始]与[思考结束]之间内容饱满，用了"首先/其次/最后"等词，结构清晰。
    [2.0 到 3.9]：格式正确，语言基本连贯。
    [0.0 到 1.9]：格式正确，但思考过程太短。
    [-2.0 到 -0.1]：有轻微复读或逻辑结巴。
    [-4.0 到 -2.1]：存在严重的复读废话、车轱辘话。
    [-5.0]：乱码，缺少任何一个标识符，或完全答非所问。"""
    else:
        rule_prompt = """该题【绝对不能思考】！标准结构必须是：[思考开始][思考结束][回答开始]回答内容[回答结束]。
    [4.0 到 5.0]：[思考开始]与[思考结束]之间完全为空！回答排版精美。
    [2.0 到 3.9]：[思考开始]与[思考结束]之间完全为空！语言连贯。
    [0.0 到 1.9]：[思考开始]与[思考结束]之间完全为空！但回答简略。
    [-2.0 到 -0.1]：有轻微复读。
    [-5.0]：只要[思考开始]与[思考结束]之间写了任何字，或者缺少标识，直接给-5.0分！"""

    sys_prompt = base_prompt + rule_prompt
    user_content = f"问题:\n{prompt}\n\n回答:\n{safe_answer}\n\n请直接输出<score>数值</score>："

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=800
        )
        content = response.choices[0].message.content.strip()

        match = re.search(r'<score>([\-\d\.]+)</score>', content)
        if match:
            score = float(match.group(1))
            return max(min(score, 5.0), -5.0)
        else:
            fallback_match = re.search(r'^\s*([\-\d\.]+)\s*$', content)
            if fallback_match:
                score = float(fallback_match.group(1))
                return max(min(score, 5.0), -5.0)

            print(f"[LLM Judge 解析失败] 返回内容:\n{content}\n" + "-" * 20)
            return 0.0

    except Exception as e:
        error_msg = str(e)

        if "1301" in error_msg or "contentFilter" in error_msg:
            return -2.0

        print(f"[LLM Judge API 错误]: {error_msg}")

        if "余额不足" in error_msg or "429" in error_msg or "1113" in error_msg:
            print("🚨 致命错误：API 余额不足或被限流！强行终止！")
            import os
            os._exit(1)

        return 0.0


def replace_spec_tokens(text: str) -> str:
    text = text.replace('<system></s>', '')
    spec_tokens = TrainerTools().tokenizer.get_special_tokens_dict().keys()
    for spec_token in spec_tokens:
        text = text.replace(spec_token, '')
    return text.strip()


def reward_func(
        prompt_ids: List[torch.Tensor],
        completion_ids: torch.Tensor,
        answers: List[Optional[torch.Tensor]]) -> List[float]:
    prompts_text = TrainerTools().tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
    completions_text = TrainerTools().tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

    batch_size = len(prompts_text)
    total_scores = [0.0] * batch_size

    rm_inputs_text = []
    rm_indices = []
    log_details = {}

    SCORE_EOS_PENALTY = -1.0
    SCORE_FORMAT_PENALTY = -1.5
    SCORE_RULE_PENALTY = -1.5
    RM_WEIGHT = 1.0

    debug_scores = {
        "eos_score": 0.0,
        "format_score": 0.0,
        "length_bonus": 0.0,
        "rm_raw": 0.0,
        "rm_weighted": 0.0
    }

    for idx, (prompt, completion) in enumerate(zip(prompts_text, completions_text)):
        completion_clean = completion.replace("<pad>", "").strip()
        has_eos = completion_clean.endswith('</s>')

        is_no_think = bool(re.search(r'/no think\s*$', prompt))
        is_think = bool(re.search(r'/think\s*$', prompt))

        ans_len = len(completion_clean)
        length_bonus = 0.0
        format_score = 0.0
        is_format_fatal = False

        if not has_eos:
            format_score += SCORE_EOS_PENALTY

        if is_think:
            if ans_len < 80:
                format_score -= 2.0
            else:
                length_bonus = min(ans_len / 400.0, 1.5)
        else:
            if ans_len < 20:
                format_score -= 2.0
            else:
                length_bonus = min(ans_len / 200.0, 1.0)

        think_content = ""
        answer_content = ""

        if completion_clean.count('<think>') != 1 or completion_clean.count('</think>') != 1 or \
                completion_clean.count('<answer>') != 1 or completion_clean.count('</answer>') != 1:

            format_score += SCORE_FORMAT_PENALTY
            is_format_fatal = True
        else:
            think_match = re.search(r'<think>(.*?)</think>', completion_clean, flags=re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', completion_clean, flags=re.DOTALL)

            if not think_match or not answer_match:
                format_score += SCORE_FORMAT_PENALTY
                is_format_fatal = True
            else:
                think_content = think_match.group(1).strip()
                answer_content = answer_match.group(1).strip()

                idx_think_end = completion_clean.find('</think>')
                idx_answer_start = completion_clean.find('<answer>')
                if idx_think_end > idx_answer_start:
                    format_score += SCORE_FORMAT_PENALTY
                    is_format_fatal = True

                if len(answer_content) == 0:
                    format_score += SCORE_FORMAT_PENALTY
                    is_format_fatal = True

                if is_no_think and len(think_content) > 0:
                    format_score += SCORE_RULE_PENALTY
                    is_format_fatal = True
                elif is_think and len(think_content) == 0:
                    format_score += SCORE_RULE_PENALTY
                    is_format_fatal = True

        current_score = format_score + length_bonus
        total_scores[idx] = current_score

        clean_prompt = replace_spec_tokens(prompt)
        clean_prompt = re.sub(r'/no think\s*$', '', clean_prompt)
        clean_prompt = re.sub(r'/think\s*$', '', clean_prompt).strip()

        clean_answer_for_rm = completion_clean

        rm_inputs_text.append((clean_prompt, clean_answer_for_rm, is_think))
        rm_indices.append((idx, is_format_fatal))

        if idx == 0:
            debug_scores["eos_score"] = 0.0 if has_eos else SCORE_EOS_PENALTY
            debug_scores["format_score"] = format_score
            debug_scores["length_bonus"] = length_bonus
            log_details = {
                "prompt_preview": clean_prompt[:100].replace('\n', ' '),
                "answer_preview": clean_answer_for_rm[:100].replace('\n', ' '),
                "has_eos": has_eos,
                "pre_rm_score": current_score,
                "is_think": is_think,
                "is_no_think": is_no_think,
                "is_format_fatal": is_format_fatal
            }

    if len(rm_inputs_text) > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(1, batch_size)) as executor:
            future_to_idx = {
                executor.submit(call_llm_judge, p, a, is_think_req): i
                for i, (p, a, is_think_req) in enumerate(rm_inputs_text)
            }

            scores = [0.0] * len(rm_inputs_text)
            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    scores[i] = future.result()
                except concurrent.futures.TimeoutError:
                    print(f"LLM Judge 请求超时 (idx: {i})，给予中性分 0.0")
                    scores[i] = 0.0
                except Exception as exc:
                    print(f"LLM Judge 多线程异常: {exc}")
                    scores[i] = 0.0

        for i, (original_idx, is_fatal) in enumerate(rm_indices):
            raw_rm_val = scores[i]
            weighted_rm_val = raw_rm_val * RM_WEIGHT
            total_scores[original_idx] += weighted_rm_val

            if original_idx == 0:
                debug_scores["rm_raw"] = raw_rm_val
                debug_scores["rm_weighted"] = weighted_rm_val

    if log_details:
        log_details["final_total"] = total_scores[0]

    if TrainerTools().parallel.is_main_process and log_details:
        with open('./log/reward.txt', 'a', encoding='utf-8') as f:
            f.write("-" * 65 + "\n")
            f.write(
                f"Prompt (Think:{log_details['is_think']}, NoThink:{log_details['is_no_think']}): {log_details['prompt_preview']}...\n")
            f.write(f"Parsed Answer: {log_details['answer_preview']}...\n")

            eos_status = "✅" if log_details['has_eos'] else "❌"
            fmt_status = "❌" if log_details['is_format_fatal'] else "✅"

            f.write(
                f"Reward: {log_details['final_total']:.4f} | "
                f"Breakdown:[EOS({eos_status}): {debug_scores['eos_score']}] + "
                f"[Format({fmt_status}): {debug_scores['format_score']}] + "
                f"[LenBonus: {debug_scores['length_bonus']:.2f}] + "
                f"[RM Raw: {debug_scores['rm_raw']:.2f} * {RM_WEIGHT} -> {debug_scores['rm_weighted']:.2f}]\n"
            )

    return total_scores


def ptx(prompt: List[torch.Tensor], answer: List[torch.Tensor]) -> List[torch.Tensor]:
    rst = []
    for p, a in zip(prompt, answer):
        ptx_data = torch.concat([p, a])
        rst.append(ptx_data)

        if TrainerTools().parallel.is_main_process:
            print(TrainerTools().tokenizer.decode(ptx_data))

    return rst


if __name__ == '__main__':
    eval_prompts = [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章'),
        get_eval_prompt('生态环境是人类的生存和发展的空间，所以人类是不是应当尽可能地去改变生态环境？'),
        get_eval_prompt('水资源主要是被工业用水消耗，我在生活中节约用水有意义吗？'),
        get_eval_prompt('作为历史初学者，我该如何开始我的历史学习之旅？'),
        get_eval_prompt('如果Python中的父类和子类分别定义在不同的文件里，怎样导入才能避免出现循环导入的问题呢？'),
        get_eval_prompt('你叫什么？'),
        get_eval_prompt('你是谁？')
    ]

    trainer = PPOTrainer(
        train_config=get_ppo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts,
        # ptx_builder=ptx,
    )

    trainer.train()
