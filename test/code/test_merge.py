#在同一运行环境下测试训练数据和测试数据的合并。
import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_methods(solution_str: str):
    """从模型输出中提取方法编号，如 '###2/4'、'2/4'。"""
    match = re.search(r"(?:###\s*)?([\d/]+)", solution_str)
    if match:
        methods = match.group(1).strip()
        return [m for m in methods.split("/") if m.isdigit()]
    return []


def evaluate_dataset_by_split(parquet_file: str, llm_agent, tokenizer):
    print(f"[Load] {parquet_file}")
    df = pd.read_parquet(parquet_file)

    sampling_agent = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=10,
        n=1,
        logprobs=0,
        ignore_eos=False,
        repetition_penalty=1.0,
    )

    # 按 split 聚合
    stats = {
        "train": {"sim_modified": [], "correct": 0, "total": 0},
        "test": {"sim_modified": [], "correct": 0, "total": 0},
    }

    def agent_choose_methods(instruction_text: str):
        outputs = llm_agent.generate([instruction_text], sampling_agent)
        return outputs[0].outputs[0].text.strip()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            extra_info = row["extra_info"]
            split = str(extra_info.get("split", "")).lower().strip()

            # 只统计 train/test
            if split not in ("train", "test"):
                continue

            instruction = row["prompt"][0]["content"]
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 1) 模型预测方法
            solution_str = agent_choose_methods(text)
            methods = extract_methods(solution_str)

            # fallback：空输出时回退到“无需进行优化”
            method_text_map = {
                "1": "无需进行优化",
                "2": "缩短指令长度",
                "3": "增加任务说明",
                "4": "调整指令结构",
            }
            if not methods:
                methods = ["1"]  # 约定 1 = 无需优化

            methods_set = set(methods)

            # 2) oracle 信息
            oracle_scores = extra_info.get("oracle_method_scores", {})
            hard_label = extra_info.get("oracle_hard_label", None)

            if hard_label is None or hard_label not in oracle_scores:
                # 无法计算准确度/分数则跳过
                continue

            best_methods = oracle_scores[hard_label].get("methods", [])
            best_methods_set = set(best_methods)

            # 准确度：是否命中最优方法集合
            is_correct = int(methods_set == best_methods_set)

            # 3) 计算 sim_modified（优先精确匹配，否则 fallback 到“无需进行优化”）
            sim_no_opt = None
            for _, info in oracle_scores.items():
                combo_texts = info.get("method_texts", [])
                if "无需进行优化" in combo_texts:
                    sim_no_opt = float(info.get("sim_score", 0.0))
                    break
            if sim_no_opt is None:
                sim_no_opt = 0.0

            sim_modified = None
            for _, info in oracle_scores.items():
                combo_methods = info.get("methods", [])
                if set(combo_methods) == methods_set:
                    sim_modified = float(info.get("sim_score", 0.0))
                    break
            if sim_modified is None:
                sim_modified = sim_no_opt

            # 4) 聚合到 split
            stats[split]["sim_modified"].append(sim_modified)
            stats[split]["correct"] += is_correct
            stats[split]["total"] += 1

        except Exception as e:
            print(f"[Warning] row {idx} failed: {e}")
            continue

    # 输出结果
    result = {}
    print("\n================ Result by split ================")
    for split in ("train", "test"):
        total = stats[split]["total"]
        mean_sim = float(np.mean(stats[split]["sim_modified"])) if total > 0 else 0.0
        acc = float(stats[split]["correct"] / total) if total > 0 else 0.0

        result[split] = {
            "mean_sim_modified": mean_sim,
            "accuracy": acc,
            "count": total,
        }

        print(
            f"[{split}] count={total}, "
            f"mean_sim_modified={mean_sim:.4f}, "
            f"accuracy={acc:.4f}"
        )
    print("=================================================\n")

    return result


if __name__ == "__main__":
    agent_model_path = "/root/autodl-tmp/verl/checkpoints/verl_grpo_sharegpt/qwen2_0.5b_fixed8/global_step_500/actor/huggingface"
    test_file_path = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/merged_oracle.parquet"

    tokenizer = AutoTokenizer.from_pretrained(
        agent_model_path,
        trust_remote_code=True
    )

    llm_agent = LLM(
        model=agent_model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
        skip_tokenizer_init=False,
        disable_log_stats=True,
        seed=42,
    )

    evaluate_dataset_by_split(test_file_path, llm_agent, tokenizer)