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
    except Exception:
        return 0.0

def generate_local_response(llm, prompt: str, sampling_params, max_new_tokens=0):
    if max_new_tokens == 0:
        max_new_tokens = len(prompt)
    sampling_params.max_tokens = max_new_tokens
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

def extract_instruction(text: str) -> str:
    match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

# =============================
# 构造方法组合（固定8种，与sharegpt_simple.py一致）
# =============================
def build_method_combinations():
    return {
        "1": {
            "methods": ["1"],
            "method_texts": ["无需进行优化"],
        },
        "2": {
            "methods": ["2"],
            "method_texts": ["缩短指令长度"],
        },
        "3": {
            "methods": ["3"],
            "method_texts": ["增加任务说明"],
        },
        "4": {
            "methods": ["4"],
            "method_texts": ["调整指令结构"],
        },
        "5": {
            "methods": ["2","3"],
            "method_texts": ["缩短指令长度","增加任务说明"],
        },
        "6": {
            "methods": ["2","4"],
            "method_texts": ["缩短指令长度","调整指令结构"],
        },
        "7": {
            "methods": ["3","4"],
            "method_texts": ["增加任务说明","调整指令结构"],
        },
        "8": {
            "methods": ["2","3","4"],
            "method_texts": ["缩短指令长度","增加任务说明","调整指令结构"],
        },
    }

# =============================
# 主处理函数
# =============================
def build_oracle_labels(
    input_parquet,
    output_parquet,
    llm,
    model_sbert,
    tokenizer,
    tau=0.1,
    debug_n=15,
):
    sampling_llm = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=128,
        n=1,
        logprobs=0,
        ignore_eos=False,
        repetition_penalty=1.0,
    )

    df = pd.read_parquet(input_parquet)
    new_rows = []

    # ===== 新增：统计变量 =====
    best_sim_sum = 0.0
    best_sim_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            extra_info = dict(row["extra_info"])
            reward_model = row["reward_model"]

            original_instruction = extra_info["instruction"]
            ground_truth = reward_model["ground_truth"]

            num_tokens_truth = len(
                tokenizer.encode(ground_truth, add_special_tokens=False)
            )

            combos = build_method_combinations()

            oracle_scores = {}

            # ===== 遍历方法组合 =====
            for combo_id, combo in combos.items():
                method_texts = combo["method_texts"]

                if any("无需进行优化" in t for t in method_texts):
                    final_instruction = original_instruction
                else:
                    modify_methods = "、".join(method_texts)
                    modify_prompt = (
                        f"你是一个指令优化器，负责优化用户给到大模型的指令，"
                        f"请在保留信息完整的前提下按照{modify_methods}的方法优化"
                        f"<instruction>和</instruction>之间的指令。"
                        f"<instruction>\n{original_instruction}\n</instruction>"
                        f"并按照“<result>你优化后的指令</result>”的格式输出结果，"
                        f"并在</result>后终止输出。"
                    )
                    messages = [
                    {"role": "user", "content":modify_prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,  # Set to False to strictly disable thinking
                    )
                    raw_modified = generate_local_response(
                        llm, text, sampling_llm
                    )
                    final_instruction = extract_instruction(raw_modified)
                
                messages2 = [
                    {"role": "user", "content":final_instruction}
                ]
                text2 = tokenizer.apply_chat_template(
                    messages2,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Set to False to strictly disable thinking
                )

                answer = generate_local_response(
                    llm, text2, sampling_llm, num_tokens_truth
                )

                sim = cosine_similarity(
                    model_sbert, answer, ground_truth
                )

                oracle_scores[combo_id] = {
                    "methods": combo["methods"],
                    "method_texts": method_texts,
                    "sim_score": float(sim),
                }

            # ===== 硬标签 =====
            best_combo = max(
                oracle_scores.items(),
                key=lambda x: x[1]["sim_score"]
            )[0]

            best_sim = oracle_scores[best_combo]["sim_score"]
            best_sim_sum += best_sim
            best_sim_count += 1

            # ===== 写回 =====
            extra_info["oracle_method_scores"] = oracle_scores
            extra_info["oracle_hard_label"] = best_combo
            row["extra_info"] = extra_info
            new_rows.append(row)

            # ===== DEBUG =====
            if idx < debug_n:
                print("\n" + "=" * 80)
                print(f"[DEBUG SAMPLE {idx}]")
                print("- 原始指令:")
                print(original_instruction)
                print("\n- oracle scores:")
                for cid, info in oracle_scores.items():
                    print(
                        f"  combo {cid}: "
                        f"methods={info['methods']} | "
                        f"sim={info['sim_score']:.4f}"
                    )
                print("\n- HARD LABEL:", best_combo)
                print("=" * 80)

        except Exception as e:
            print(f"[Warning] row {idx} failed: {e}")
            continue

    new_df = pd.DataFrame(new_rows)
    new_df.to_parquet(output_parquet, index=False)

    # ===== 最终统计输出 =====
    mean_best_sim = best_sim_sum / best_sim_count if best_sim_count > 0 else 0.0

    print(f"[Done] Saved oracle dataset to {output_parquet}")
    print(f"[Stats] Best sim_count  = {best_sim_count}")
    print(f"[Stats] Best sim_score sum  = {best_sim_sum:.6f}")
    print(f"[Stats] Best sim_score mean = {mean_best_sim:.6f}")

# =============================
# 主入口
# =============================
if __name__ == "__main__":
    model_sbert = SentenceTransformer(
        "/root/autodl-tmp/model/all-mpnet-base-v2",
        device="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/model/qwen3-8b",
        trust_remote_code=True
    )

    llm = LLM(
        model="/root/autodl-tmp/model/qwen3-8b",
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=2048,
        skip_tokenizer_init=False,
        disable_log_stats=True,
        seed=42,
    )

    build_oracle_labels(
        input_parquet="/root/autodl-tmp/data/processed_sharegpt/fixed_8/test.parquet",
        output_parquet="/root/autodl-tmp/data/processed_sharegpt/fixed_8/test_oracle.parquet",
        llm=llm,
        model_sbert=model_sbert,
        tokenizer=tokenizer,
        tau=0.1,
    )
