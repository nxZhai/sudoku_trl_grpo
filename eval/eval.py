import numpy as np
import re
import json
import tqdm

"""
1. 原模型和训练后的模型对比
2. 使用vllm跑推理，大概300条数据，统计准确度
3. 使用和训练相同的提示词
"""


## 奖励函数对应的分数
# 格式奖励
def format_score(completion):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    matches = [re.match(pattern, content, re.DOTALL) for content in [completion]]
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
    return rewards


## 问题描述准确奖励
def question_match_reward(completion, question, **kwargs):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    matches = [re.match(pattern, content, re.DOTALL) for content in [completion]]
    rewards = []
    for match, qs_str in zip(matches, [question]):
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

    return rewards


def ans_row_score(completion):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    matches = [re.match(pattern, content, re.DOTALL) for content in [completion]]
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
    return rewards


def ans_col_score(completion):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    matches = [re.match(pattern, content, re.DOTALL) for content in [completion]]
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
    return rewards


def ans_block_reward(completion):
    pattern = r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
    matches = [re.match(pattern, content, re.DOTALL) for content in [completion]]
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
    return rewards


## 整体的统计准确度的函数
def calculate_score(data_path, result_path):
    # 读取jsonl数据
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    # 计算各项奖励的均值
    for data in tqdm.tqdm(all_data, desc="Calculating scores"):
        completions = data["completion"]
        question = data["question"]

        # 奖励计算
        format_rewards = format_score(completions)
        question_rewards = question_match_reward(completions, question)
        row_rewards = ans_row_score(completions)
        col_rewards = ans_col_score(completions)
        block_rewards = ans_block_reward(completions)

        # 保存各项奖励到数据中
        data["format_rewards"] = format_rewards
        data["question_rewards"] = question_rewards
        data["row_rewards"] = row_rewards
        data["col_rewards"] = col_rewards
        data["block_rewards"] = block_rewards

        ###############

        # 正确分数计算
        format_scores = [1 if f == 0.6 else 0 for f in format_rewards]
        question_scores = [1 if q == 1.0 else 0 for q in question_rewards]
        row_scores = [1 if r == 0.4 else 0 for r in row_rewards]
        col_scores = [1 if c == 0.3 else 0 for c in col_rewards]
        block_scores = [1 if b == 0.3 else 0 for b in block_rewards]
        total_scores = [
            1 if (r + c + b) == 1.0 else 0
            for r, c, b in zip(row_rewards, col_rewards, block_rewards)
        ]

        format_exclude = [
            1 if (r + c + b) == 1.0 and f == 0.6 else 0
            for f, r, c, b in zip(
                format_rewards, row_rewards, col_rewards, block_rewards
            )
        ]

        # 保存score分数
        data["format_scores"] = format_scores
        data["question_scores"] = question_scores
        data["row_scores"] = row_scores
        data["col_scores"] = col_scores
        data["block_scores"] = block_scores
        data["total_scores"] = total_scores
        data["format_exclude"] = format_exclude

    ## 计算各项奖励的均值
    format_mean = np.mean([np.mean(data["format_rewards"]) for data in all_data])
    question_mean = np.mean([np.mean(data["question_rewards"]) for data in all_data])
    row_mean = np.mean([np.mean(data["row_rewards"]) for data in all_data])
    col_mean = np.mean([np.mean(data["col_rewards"]) for data in all_data])
    block_mean = np.mean([np.mean(data["block_rewards"]) for data in all_data])

    # 求百分比
    format_mean_pct = format_mean / 0.6
    question_mean_pct = question_mean / 1.0
    row_mean_pct = row_mean / 0.4
    col_mean_pct = col_mean / 0.3
    block_mean_pct = block_mean / 0.3

    ## 总共对了多少个
    prompt_score = np.sum([np.sum(data["format_scores"]) for data in all_data])
    question_score = np.sum([np.sum(data["question_scores"]) for data in all_data])
    row_score = np.sum([np.sum(data["row_scores"]) for data in all_data])
    col_score = np.sum([np.sum(data["col_scores"]) for data in all_data])
    block_score = np.sum([np.sum(data["block_scores"]) for data in all_data])
    total_score = np.sum([np.sum(data["total_scores"]) for data in all_data])

    format_exclude = np.sum([np.sum(data["format_exclude"]) for data in all_data])

    prompt_score_pct = prompt_score / len(all_data)
    question_score_pct = question_score / len(all_data)
    row_score_pct = row_score / len(all_data)
    col_score_pct = col_score / len(all_data)
    block_score_pct = block_score / len(all_data)
    total_score_pct = total_score / len(all_data)

    format_exclude_pct = (
        format_exclude / len(all_data) / prompt_score_pct
        if prompt_score_pct > 0
        else 0.0
    )

    # 将结果保存到一个json文件中
    result = {
        "mean_scores": {
            "format_mean": format_mean_pct,
            "question_mean": question_mean_pct,
            "row_mean": row_mean_pct,
            "col_mean": col_mean_pct,
            "block_mean": block_mean_pct,
        },
        "num_score": {
            "prompt_score": prompt_score_pct,
            "question_score": question_score_pct,
            "row_score": row_score_pct,
            "col_score": col_score_pct,
            "block_score": block_score_pct,
            "total_score": total_score_pct,
            "format_exclude": format_exclude_pct,
        },
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print("Evaluation results saved to", result_path)
    return result


if __name__ == "__main__":
    import os

    os.makedirs("./score", exist_ok=True)

    data_path = "/data/zhainx/swanlab/sudoku_trl_grpo/eval/data/qwen-7b-chat-temperature0-original.jsonl"
    result_path = "./score/qwen-7b-chat-temperature0-original-score.json"

    calculate_score(data_path, result_path)
