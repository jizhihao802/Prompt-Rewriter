#指令修改和回答仅执行一次，并且选项全部打乱
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# =============================
# 工具函数
# =============================
def embed(model_sbert, text: str):
    return model_sbert.encode(text, normalize_embeddings=True)

def cosine_similarity(model_sbert, a: str, b: str) -> float:
    try:
        return float(util.cos_sim(embed(model_sbert, a), embed(model_sbert, b)))
    except Exception as e:
        print(f"[Warning] cosine_similarity failed: {e}")
        return 0.0

def generate_local_response(llm, prompt: str, sampling_params, max_new_tokens=0):
    if max_new_tokens == 0:
        max_new_tokens = len(prompt)
    sampling_params.max_tokens = max_new_tokens
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

def extract_methods(solution_str: str):
    match = re.search(r"(?:###\s*)?([\d/]+)", solution_str)
    if match:
        return [m for m in match.group(1).strip().split("/") if m.isdigit()]
    return []

def extract_instruction(text: str) -> str:
    match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# =============================
# 测试函数
# =============================
def evaluate_parquet_test(test_file, llm, llm_agent, model_sbert, tokenizer):
    """
    测试数据 parquet 文件，每行 dict 包含：
        prompt -> [{"role": "user", "content": instruction}]
        reward_model -> {"ground_truth": ...}
        extra_info -> {"orig_similarity": ..., "baseline_similarity": ...}
    输出三个均值：
        1. modified
        2. baseline
        3. original
    """
    log_path = "/root/autodl-tmp/test/sft_test3/log.txt"
    with open(log_path, "w") as f:
        f.write("modified,baseline,original\n")

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
    sampling_llm = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=128,
        n=1,
        logprobs=0,
        ignore_eos=False,
        repetition_penalty=1.0,
    )

    def agent_choose_methods(instruction):
        out = llm_agent.generate([instruction], sampling_agent)
        return out[0].outputs[0].text.strip()

    # 读取 parquet 测试文件
    df = pd.read_parquet(test_file)

    sim_modified_list = []
    sim_baseline_list = []
    sim_original_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            extra_info = row["extra_info"]
            reward_model = row["reward_model"]

            # 指令修改选择
            instruction = row["prompt"][0]["content"]

            #原指令
            original_instruction = extra_info["instruction"]

            ground_truth = reward_model["ground_truth"]
            orig_sim = float(extra_info.get("orig_similarity", 0.0))
            baseline_sim = float(extra_info.get("baseline_similarity", 0.0))

            num_tokens_truth = len(tokenizer.encode(ground_truth, add_special_tokens=False))

            full_instruction = (
                f"system\n"
                f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
                f"user\n"
                f"{instruction}\n"
                f"assistant\n"
            )

            #print(full_instruction)

            # 1. agent 选择方法
            solution_str = agent_choose_methods(full_instruction)
            methods = extract_methods(solution_str)
            print(solution_str)

            with open(log_path, "a") as f:
                f.write(f"{solution_str}\n{methods}\n")

            if not methods:
                methods = []

            # ===== 2. 根据 method_mapping 决定是否修改 =====
            method_text_map = extra_info.get("method_mapping", {})
            with open(log_path, "a") as f:
                f.write(f"{method_text_map}\n")
            selected_texts = [method_text_map.get(m, "") for m in methods]
            selected_texts = [t for t in selected_texts if t]

            # 是否需要修改
            if not selected_texts or "无需进行优化" in selected_texts:
                final_instruction = original_instruction
                with open(log_path, "a") as f:
                  f.write("无需进行优化\n")
            else:
                modify_methods = "、".join(selected_texts)
                with open(log_path, "a") as f:
                  f.write(f"{modify_methods}\n")
                modify_prompt = (
                    f"你是一个指令优化器，负责优化用户给到大模型的指令，"
                    f"请在保留信息完整的前提下按照{modify_methods}的方法优化"
                    f"<instruction>和</instruction>之间的指令。"
                    f"<instruction>\n{original_instruction}\n</instruction>"
                    f"并按照“<result>你优化后的指令</result>”的格式输出结果，"
                    f"并在</result>后终止输出。"
                )

                raw_modified = generate_local_response(
                    llm, modify_prompt, sampling_llm
                )
                final_instruction = extract_instruction(raw_modified)

            # ===== 3. 用最终 instruction 生成回答 =====
            new_answer = generate_local_response(
                llm, final_instruction, sampling_llm, num_tokens_truth
            )
            sim_modified = cosine_similarity(model_sbert, new_answer, ground_truth)

            # baseline
            sim_baseline = baseline_sim

            # 原始指令回答
            sim_original = orig_sim

            print(sim_modified, sim_baseline, sim_original)

            # ========== 新增：实时写入文件 ==========
            with open(log_path, "a") as f:
                f.write(f"{sim_modified},{sim_baseline},{sim_original}\n")

            # 记录
            sim_modified_list.append(sim_modified)
            sim_baseline_list.append(sim_baseline)
            sim_original_list.append(sim_original)

        except Exception as e:
            print(f"[Warning] Failed processing row {idx}: {e}")
            continue

    # 输出均值
    print("\n============== Final Evaluation ==============")
    print(f"修改后相似度均值 (modified): {np.mean(sim_modified_list):.4f}")
    print(f"baseline 相似度均值        : {np.mean(sim_baseline_list):.4f}")
    print(f"原始指令相似度均值 (orig)  : {np.mean(sim_original_list):.4f}")
    print("==============================================")

    return {
        "modified_avg": float(np.mean(sim_modified_list)),
        "baseline_avg": float(np.mean(sim_baseline_list)),
        "original_avg": float(np.mean(sim_original_list)),
    }


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    # 设备
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"[Init] device: {device}")

    # Sentence-BERT
    model_sbert = SentenceTransformer(
        "/root/autodl-tmp/model/all-mpnet-base-v2",
        device="cpu"
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/model/qwen3-8b",
        trust_remote_code=True
    )

    # 回答模型
    llm = LLM(
        model="/root/autodl-tmp/model/qwen3-8b",
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization = 0.5,
        skip_tokenizer_init=False,
        disable_log_stats=True,
        seed=42,
    )

    # agent 模型（选择方法）
    agent_model_path = "/root/autodl-tmp/model/qwen2.5-1.5b"
    llm_agent = LLM(
        model=agent_model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization = 0.2,
        skip_tokenizer_init=False,
        disable_log_stats=True,
        seed=42,
    )

    # 测试文件路径
    test_file_path = "/root/autodl-tmp/data/processed_sharegpt/test/test2_shuffle2.parquet"

    # 调用测试
    evaluate_parquet_test(test_file_path, llm, llm_agent, model_sbert, tokenizer)