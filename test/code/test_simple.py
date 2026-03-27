import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =============================
# 工具函数
# =============================

def extract_methods(solution_str: str):
    """
    从模型输出中提取优化方法编号
    支持：
      - '###2/4'
      - '2/4'
      - '选择了方法 2/4'
    返回列表形式，例如 ['2','4']
    """
    match = re.search(r"(?:###\s*)?([\d/]+)", solution_str)
    if match:
        methods = match.group(1).strip()
        return [m for m in methods.split("/") if m.isdigit()]
    return []


# =============================
# 核心测试函数（纯查表版）
# =============================

def evaluate_parquet_test(test_file, llm_agent, tokenizer=None):

    log_path = "/root/autodl-tmp/test/model8/log0.txt"
    with open(log_path, "w") as f:
        f.write("modified\n")

    print(f"[Load] Reading test file: {test_file}")
    df = pd.read_parquet(test_file)

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

    def agent_choose_methods(instruction):
        outputs = llm_agent.generate([instruction], sampling_agent)
        return outputs[0].outputs[0].text.strip()

    sim_modified_list = []
    sim_no_opt_list = []   # ⭐ 新增：统计“无需优化”sim
    hit_best = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:
            extra_info = row["extra_info"]
            instruction = row["prompt"][0]["content"]

            #full_instruction = (
            #    f"system\n"
            #    f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
            #    f"user\n"
            #    f"{instruction}\n"
            #    f"assistant\n"
            #)
            messages = [
				{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # ===== 1️⃣ agent 选择方法 =====
            solution_str = agent_choose_methods(text)
            methods = extract_methods(solution_str)

            with open(log_path, "a") as f:
                f.write(f"{solution_str}\n{methods}\n")

            #method_text_map = extra_info.get("method_mapping", {})
            method_text_map = {
                "1": "无需进行优化",
                "2": "缩短指令长度",
                "3": "增加任务说明",
                "4": "调整指令结构",
            }
            with open(log_path, "a") as f:
                f.write(f"{method_text_map}\n")

            # 若模型未输出方法 → 自动填充“无需进行优化”
            if not methods:
                for k, v in method_text_map.items():
                    if str(v).strip() == "无需进行优化":
                        methods = [str(k)]
                        print(f"[Auto-Fill] methods为空，自动填充为无需优化编号: {methods}")
                        break

            selected_texts = [method_text_map.get(m, "") for m in methods]
            selected_texts = [t for t in selected_texts if t]

            if not selected_texts or "无需进行优化" in selected_texts:
                with open(log_path, "a") as f:
                    f.write("无需进行优化\n")
            else:
                modify_methods = "、".join(selected_texts)
                with open(log_path, "a") as f:
                    f.write(f"{modify_methods}\n")

            print("\n" + "=" * 60)
            print(f"[Sample {idx}]")
            print("Agent output:", solution_str)
            print("Parsed methods:", methods)

            oracle_scores = extra_info.get("oracle_method_scores", {})
            methods_set = set(methods)

            #提取最优解
            hard_label = extra_info["oracle_hard_label"]
            best_methods = oracle_scores[hard_label]["methods"]

            if set(best_methods) == methods_set:
                hit_best += 1

            # =============================
            # ⭐ 统计“无需进行优化”的 sim_score
            # =============================
            sim_no_opt = None
            for combo_id, info in oracle_scores.items():
                combo_texts = info.get("method_texts", [])
                if "无需进行优化" in combo_texts:
                    sim_no_opt = float(info.get("sim_score", 0.0))
                    break

            if sim_no_opt is None:
                sim_no_opt = 0.0

            sim_no_opt_list.append(sim_no_opt)

            # =============================
            # 正常匹配 agent 输出
            # =============================
            sim_modified = None

            # 1️⃣ 尝试精确匹配
            for combo_id, info in oracle_scores.items():
                combo_methods = info.get("methods", [])
                if set(combo_methods) == methods_set:
                    sim_modified = float(info.get("sim_score", 0.0))
                    print(
                        f"✅ HIT oracle combo={combo_id} "
                        f"methods={combo_methods} "
                        f"sim={sim_modified:.4f}"
                    )
                    break

            # 2️⃣ 未匹配成功 → fallback 到 “无需进行优化”
            if sim_modified is None:
                print("⚠️ No matching combo → fallback to '无需进行优化'")
                sim_modified = sim_no_opt

            print("modified:", sim_modified)
            print("no_opt :", sim_no_opt)

            sim_modified_list.append(sim_modified)

        except Exception as e:
            print(f"[Warning] Failed row {idx}: {e}")
            continue

    # =============================
    # 统计结果
    # =============================

    print("\n============== Final Evaluation ==============")

    modified_avg = np.mean(sim_modified_list) if sim_modified_list else 0.0
    no_opt_avg = np.mean(sim_no_opt_list) if sim_no_opt_list else 0.0
    hit_best_rate = hit_best / len(df) if len(df) > 0 else 0.0

    print(f"修改后相似度均值 (modified): {modified_avg:.4f}")
    print(f"无需优化相似度均值 (no_opt): {no_opt_avg:.4f}")
    print(f"命中最优解率: {hit_best_rate:.4f}")
    print("==============================================")

    return {
        "modified_avg": float(modified_avg),
        "no_opt_avg": float(no_opt_avg),
    }


# =============================
# 主程序入口
# =============================

if __name__ == "__main__":

    #agent_model_path = "/root/autodl-tmp/verl/checkpoints/verl_grpo_sharegpt/qwen2_0.5b_add/global_step_520/actor/huggingface"
    #agent_model_path = "/root/autodl-tmp/verl/checkpoints/verl_grpo_sharegpt/qwen2_0.5b_fixed8/global_step_500/actor/huggingface"
    agent_model_path = "/root/autodl-tmp/model/qwen2.5-0.5b-soft-sft/checkpoint-1880"
    print("[Init] Loading agent model...")

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

    #test_file_path = "/root/autodl-tmp/data/processed_sharegpt/test/test2_shuffle2_simple.parquet"
    test_file_path = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/test_oracle.parquet"
    evaluate_parquet_test(test_file_path, llm_agent, tokenizer)