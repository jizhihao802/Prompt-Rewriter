#测试原始指令所得回答
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
    log_path = "/root/autodl-tmp/test/log1.txt"
    with open(log_path, "w") as f:
        f.write("original\n")

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

            #原指令
            original_instruction = extra_info["instruction"]
            ground_truth = reward_model["ground_truth"]

            num_tokens_truth = len(tokenizer.encode(ground_truth, add_special_tokens=False))

            final_instruction = original_instruction

            # ===== 3. 用最终 instruction 生成回答 =====
            new_answer = generate_local_response(
                llm, final_instruction, sampling_llm, num_tokens_truth
            )
            sim_original = cosine_similarity(model_sbert, new_answer, ground_truth)

            print(sim_original)

            # ========== 新增：实时写入文件 ==========
            with open(log_path, "a") as f:
                f.write(f"{sim_original}\n")

            # 记录
            sim_original_list.append(sim_original)

        except Exception as e:
            print(f"[Warning] Failed processing row {idx}: {e}")
            continue

    # 输出均值
    print("\n============== Final Evaluation ==============")
    print(f"原始指令相似度均值 (orig)  : {np.mean(sim_original_list):.4f}")
    print("==============================================")

    return {
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
    agent_model_path = "/root/autodl-tmp/verl/checkpoints/verl_examples/sharegpt4/global_step_80/actor/huggingface"
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
    test_file_path = "/root/autodl-tmp/data/processed_sharegpt/test/test2_shuffle3.parquet"

    # 调用测试
    evaluate_parquet_test(test_file_path, llm, llm_agent, model_sbert, tokenizer)