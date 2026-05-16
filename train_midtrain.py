from llm_trainer import Trainer
from utils import init_env, get_midtrain_config

if __name__ == '__main__':
    init_env()
    eval_prompts = [
        '自来水直接冲洗生肉不仅冲不掉细菌',
    ]

    trainer = Trainer(
        train_config=get_midtrain_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()