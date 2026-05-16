from llm_trainer import DPOTrainer
from utils import init_env, get_dpo_config, get_eval_prompt

if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章'),
        get_eval_prompt('生态环境是人类的生存和发展的空间，所以人类是不是应当尽可能地去改变生态环境？'),
        get_eval_prompt('水资源主要是被工业用水消耗，我在生活中节约用水有意义吗？'),
        get_eval_prompt('作为历史初学者，我该如何开始我的历史学习之旅？'),
        get_eval_prompt('如果Python中的父类和子类分别定义在不同的文件里，怎样导入才能避免出现循环导入的问题呢？'),
        get_eval_prompt('你叫什么？'),
        get_eval_prompt('你是谁？')
    ]

    trainer = DPOTrainer(
        train_config=get_dpo_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()