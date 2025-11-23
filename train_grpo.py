import os
import re
from dataclasses import dataclass

import datasets
from trl import ModelConfig, GRPOConfig, GRPOTrainer, TrlParser, get_peft_config

import swanlab
import numpy as np


################################################
# 自定义参数类
################################################
rl_setting = swanlab.Settings(max_log_length=4096)
swanlab.merge_settings(rl_setting)


################################################
# 自定义参数类
################################################
@dataclass
class DatasetArguments:
    """数据集参数的数据类"""

    dataset_id_or_path: str = "json"
    data_files: str = "./data/sudoku_4x4_qa_nostar.jsonl"


################################################
# 提示词格式
################################################
PROMPT_TEMPLATE = """请作为数独专家完成4x4数独题目。数独题目中已给出数字为限定条件，“_”表示待填入1-4的空白位置。
## 数独规则
- 根据数独矩阵中已给出的数字推测“_”处的数字
- 每行：数字1-4各出现一次
- 每列：数字1-4各出现一次  
- 每个2×2宫格：数字1-4各出现一次
## 解题流程
1. 选择一处空白位置“_”，按行、列、宫格分析应该填入的数字
2. 打印填补一处空白的矩阵，判断该次填充是否正确
3. 重复流程1，直至所有空白位置“_”被填完。
## 输出格式
输出格式为<think>...</think>\n<answer>...</answer>
在<think>...</think>中填入思考过程
在<answer>...</answer>中填写python List格式的最终矩阵
输出格式案例：
<think>
详细推理过程，包括：
- 一步步的思考过程
- 对题目的复述，以及要分析的空白位置
- 填入空白位置的数字以及理由，按行、列、宫格规则分析
- 检查填入后的数独矩阵，确保符合数独规则，确保没有修改题目限定条件
</think>
<answer>[[1,2,3,4],[4,3,...],...]</answer>
## 数独题目：
{}"""

SUDOKU_FORMAT = "[[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}]]"
# SUDOKU_FORMAT = r"""
# \[ \begin{{bmatrix}}
# {} & {} & {} & {} \\
# {} & {} & {} & {} \\
# {} & {} & {} & {} \\
# {} & {} & {} & {} \\
# \end{{bmatrix}}
# \]
# """

################################################
# 奖励函数
################################################


## 格式准确奖励
def format_reward_func(completions, prompts, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for match in matches:
        if match:
            try:
                numberstr = re.findall(r"\d+", match.group(1))
                number = list(map(int, numberstr))
                # 判断数字是16个，且都在1-4
                if len(number) == 16 and all(0 < x < 5 for x in number):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    rewards = [r * 0.6 for r in rewards]

    # 打印第一组
    prompt_content = prompts[0][0]["content"]
    print("#" * 20, " 打印推理案例 ", "#" * 20)
    print(prompt_content[-111:], "\n模型回复：\n", completions[0][0]["content"])
    print("格式准确奖励：", rewards[0])

    return rewards


## 长度奖励
def len_reward_func(completions, **kwargs):
    """Reward function for response len."""
    content_len = [len(completion[0]["content"]) for completion in completions]
    rewards = []
    for clen in content_len:
        if clen < 1648 - 64:  # 2048-400=1648
            rewards.append(1.0)
        else:
            rw = (1648 - clen) / 64
            rewards.append(rw if rw >= 0 else 0.0)
    rewards = [r * 0.4 for r in rewards]

    # 打印第一组
    print("长度奖励：", rewards[0])

    return rewards


## 问题描述准确奖励
def question_match_reward(completions, question, **kwargs):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for match, qs_str in zip(matches, question):
        if match:
            try:
                numberstr = re.findall(r"\d+", match.group(1))
                question_match_num = sum(
                    1 for a, b in zip(numberstr, qs_str) if b != "_" and a == b
                )
                question_match_rate = question_match_num / (16 - qs_str.count("_"))
                rewards.append(question_match_rate)
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    rewards = [r * 1.0 for r in rewards]

    # 打印第一组
    print("问题描述准确奖励：", rewards[0])

    return rewards


## 数独行准确评估
def ans_row_reward(completions, answer, **kwargs):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for match in matches:
        if match:
            try:
                numberstr = re.findall(r"\d+", match.group(1))
                number = list(map(int, numberstr))
                if len(number) != 16:
                    rewards.append(0.0)
                else:
                    matrix_4x4 = np.array(number).reshape(4, 4)
                    reward = [len(np.unique(row)) / 4 for row in matrix_4x4]
                    rewards.append(sum(reward) / 4)
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    rewards = [r * 0.4 for r in rewards]

    # 打印第一组
    print("数独行准确评估：", rewards[0])

    return rewards


## 数独列准确评估
def ans_col_reward(completions, answer, **kwargs):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for match in matches:
        if match:
            try:
                numberstr = re.findall(r"\d+", match.group(1))
                number = list(map(int, numberstr))
                if len(number) != 16:
                    rewards.append(0.0)
                else:
                    matrix_4x4 = np.array(number).reshape(4, 4)
                    matrix_4x4 = matrix_4x4.T
                    reward = [len(np.unique(row)) / 4 for row in matrix_4x4]
                    rewards.append(sum(reward) / 4)
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    rewards = [r * 0.3 for r in rewards]

    # 打印第一组
    print("数独列准确评估：", rewards[0])

    return rewards


## 数独块准确评估
def ans_block_reward(completions, answer, **kwargs):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = []
    for match in matches:
        if match:
            try:
                numberstr = re.findall(r"\d+", match.group(1))
                number = list(map(int, numberstr))
                if len(number) != 16:
                    rewards.append(0.0)
                else:
                    matrix_4x4 = np.array(number).reshape(4, 4)
                    reward = 0
                    reward += len(np.unique(matrix_4x4[0:2, 0:2])) / 4
                    reward += len(np.unique(matrix_4x4[0:2, 2:4])) / 4
                    reward += len(np.unique(matrix_4x4[2:4, 0:2])) / 4
                    reward += len(np.unique(matrix_4x4[2:4, 2:4])) / 4
                    rewards.append(reward / 4)
            except:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    rewards = [r * 0.3 for r in rewards]

    # 打印第一组
    print("数独块准确评估：", rewards[0])

    return rewards


################################################
# 基于trl实现GRPO训练过程
################################################
def grpo_function(
    model_args: ModelConfig, dataset_args: DatasetArguments, training_args: GRPOConfig
):
    ################################################
    # 读取数据 & 加载数据
    ################################################
    sudoku_dataset = datasets.load_dataset(
        dataset_args.dataset_id_or_path, data_files=dataset_args.data_files
    )
    sudoku_dataset = sudoku_dataset.filter(lambda x: x["label"] == "simple")
    sudoku_dataset = sudoku_dataset.shuffle(seed=42)  # 固定seed打乱
    # train_dataset,eval_dataset = sudoku_dataset["train"].train_test_split(test_size=0.1, seed=42)

    split_dataset = sudoku_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    def make_conversation(example):
        sudo_str = SUDOKU_FORMAT.format(*[c for c in example["question"]])
        user_prompt = PROMPT_TEMPLATE.format(sudo_str)
        return {
            "prompt": [
                {"role": "user", "content": user_prompt},
            ],
            "answer": [int(c) for c in example["answer"]],
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    ################################################
    # 设置 GRPOTrainer & 开启训练
    ################################################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,  # 模型名称或路径
        # 奖励函数列表，用于计算奖励分数
        reward_funcs=[
            format_reward_func,
            len_reward_func,
            question_match_reward,
            ans_row_reward,
            ans_col_reward,
            ans_block_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ################################################
    # 保存训练结果
    ################################################
    trainer.save_model(training_args.output_dir)
    print(f"Model and Tokenizer saved to {training_args.output_dir}")
    print("*** Training complete! ***")


if __name__ == "__main__":
    """主函数，用于执行主训练循环"""
    # 解析命令行参数和配置文件
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig))
    args = parser.parse_args_and_config()
    # 运行主训练循环
    grpo_function(*args)
