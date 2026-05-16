import os
import json
import copy
import traceback
import torch
from bottle import Bottle, request, response, run

from llm_model import LlmModel, ModelConfig, RoPEConfig, MoEConfig
from llm_trainer import streaming_generate, TrainerTools


def get_model_config(long_context=False):
    max_position_embeddings = 2048 if long_context else 512
    original_max_position_embeddings = 512 if long_context else None
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,

        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=4,

        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        use_qk_norm=True,
        attention_implementation='sdpa',

        moe_config=MoEConfig(
            n_dense_layer=1,
            intermediate_size=512,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=2,
            norm_topk_prob=True,
            seq_aux=True,
            routed_scaling_factor=1.0,
            aux_loss_coef=1e-3,
            z_loss_coef=1e-4,
        ),

        rope_config=RoPEConfig(
            rope_type=rope_type,
            rope_theta=10000.0,
        ),
    )


def init_env():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKEN_DIR'] = './tokenizer'
    os.environ['LOG_DIR'] = './log/'

    os.environ['CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    os.environ['SAVE_BEST_CHECKPOINT'] = '0'  # or '1'


model_dir = './'
os.makedirs(model_dir, exist_ok=True)
model_name = 'dpo.bin'

if not os.path.exists(f'{model_dir}{model_name}'):
    from modelscope import snapshot_download

    snapshot_download(
        'qibin0506/Cortex-3.1',
        allow_file_pattern=[model_name],
        local_dir=model_dir
    )

app = Bottle()
init_env()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"


model = LlmModel(get_model_config(long_context=True)).to(device=device)
model.load_state_dict(torch.load(f'{model_dir}{model_name}', map_location=device, weights_only=True))
model.eval()

tokenizer = TrainerTools().tokenizer

system_tokens = tokenizer.encode("<system></s>")
max_user_tokens = 1024
max_new_tokens = 2048
ENABLE_MULTI_TURN = False

html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html')
if not os.path.exists(html_path):
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()


@app.route('/')
def index():
    response.content_type = 'text/html; charset=utf-8'
    return html


@app.route('/chat', method=['POST', 'OPTIONS'])
def chat_stream():
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return {}

    response.content_type = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['Access-Control-Allow-Origin'] = '*'

    try:
        body = request.body.read().decode('utf-8')
        payload = json.loads(body) if body else {}
    except Exception as e:
        traceback.print_exc()
        return f'data: {json.dumps({"delta": f"解析失败: {e}"}, ensure_ascii=False)}\n\n'

    chat_history = payload.get('messages', payload.get('history', []))
    content = payload.get('content', '')

    if not chat_history and content:
        chat_history = [{'role': 'user', 'content': content}]

    if not chat_history:
        return f'data: {json.dumps({"delta": "对话历史不能为空"}, ensure_ascii=False)}\n\n'

    if not ENABLE_MULTI_TURN:
        chat_history = [chat_history[-1]]

    is_think = payload.get('is_think', payload.get('thinking', True))
    temperature = float(payload.get('temperature', 0.7))
    top_p = float(payload.get('top_p', 0.9))
    top_k = int(payload.get('top_k', 40))

    def generate():
        try:
            msgs = copy.deepcopy(chat_history)

            msgs.reverse()
            chat_tokens = []

            for i, chat in enumerate(msgs):
                role = '<user>' if chat['role'] == 'user' else '<assistant>'
                chat_content = chat['content']

                if role == '<user>' and i == 0:
                    if is_think:
                        chat_content += " /think"
                    else:
                        chat_content += " /no think"

                chat_item_tokens = tokenizer.encode(f"{role}{chat_content}</s>")

                current_tokens_len = sum(len(c) for c in chat_tokens)
                if len(system_tokens) + current_tokens_len + len(chat_item_tokens) >= max_user_tokens:
                    break
                chat_tokens.append(chat_item_tokens)

            chat_tokens.reverse()
            chat_tokens = [item for sublist in chat_tokens for item in sublist]

            chat_tokens.append(tokenizer.assistant)
            chat_tokens = system_tokens + chat_tokens

            suffix = '<think>' if is_think else '<think></think><answer>'
            chat_tokens.extend(tokenizer.encode(suffix))

            print("当前生成的Prompt文本 (供调试):")
            print(tokenizer.decode(torch.tensor(chat_tokens)))

            generator = streaming_generate(
                model=model,
                prompt=torch.tensor(chat_tokens),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device,
                return_token=True
            )

            resp_tokens = []
            prev_text = ""

            for chunk in generator:
                if chunk == tokenizer.end:
                    break

                resp_tokens.append(chunk)

                current_text = tokenizer.decode(torch.tensor(resp_tokens))
                delta_text = current_text[len(prev_text):]
                prev_text = current_text

                if delta_text:
                    sse_payload = json.dumps({"delta": delta_text}, ensure_ascii=False)
                    yield f"data: {sse_payload}\n\n"

            yield f"data: [DONE]\n\n"

        except Exception as e:
            traceback.print_exc()
            error_msg = f"\n\n**[模型执行错误]**: `{str(e)}`"
            error_payload = json.dumps({"delta": error_msg}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"
            yield f"data: [DONE]\n\n"

    return generate()


if __name__ == '__main__':
    print("=" * 50)
    print("模型加载完成，多轮对话服务即将启动...")
    print(f"多轮对话模式: {'开启' if ENABLE_MULTI_TURN else '关闭'}")
    print("请打开浏览器访问： http://127.0.0.1:8080")
    print("=" * 50)

    try:
        import paste

        run(app, host='0.0.0.0', port=8080, server='paste')
    except ImportError:
        print("未检测到 paste 模块，将使用 Bottle 默认服务器启动...")
        run(app, host='0.0.0.0', port=8080)
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"