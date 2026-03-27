import itertools
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
# 构造方法组合
# =============================
def build_method_combinations(method_text_map):
    no_opt_ids = [
        k for k, v in method_text_map.items()
        if "无需进行优化" in v
    ]

    normal_ids = [
        k for k in method_text_map
        if k not in no_opt_ids
    ]

    combos = {}
    combo_idx = 0

    # 不修改
    if no_opt_ids:
        mid = no_opt_ids[0]
        combos[str(combo_idx)] = {
            "methods": [mid],
            "method_texts": [method_text_map[mid]]
        }
        combo_idx += 1

    # 其余方法子集
    for r in range(1, len(normal_ids) + 1):
        for subset in itertools.combinations(normal_ids, r):
            combos[str(combo_idx)] = {
                "methods": list(subset),
                "method_texts": [method_text_map[m] for m in subset]
            }
            combo_idx += 1

    return combos

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
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
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

            method_text_map = extra_info.get("method_mapping", {})
            combos = build_method_combinations(method_text_map)

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
                    raw_modified = generate_local_response(
                        llm, modify_prompt, sampling_llm
                    )
                    final_instruction = extract_instruction(raw_modified)

                answer = generate_local_response(
                    llm, final_instruction, sampling_llm, num_tokens_truth
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

            # ===== 软标签 =====
            combo_ids = list(oracle_scores.keys())
            scores = np.array([
                oracle_scores[c]["sim_score"]
                for c in combo_ids
            ])
            probs = np.exp(scores / tau)
            probs = probs / probs.sum()

            soft_label = {
                cid: float(p)
                for cid, p in zip(combo_ids, probs)
            }

            # ===== 写回 =====
            extra_info["oracle_method_scores"] = oracle_scores
            extra_info["oracle_hard_label"] = best_combo
            extra_info["oracle_soft_label"] = soft_label
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
        input_parquet="/root/autodl-tmp/data/processed_sharegpt/add/train_shuffle.parquet",
        output_parquet="/root/autodl-tmp/data/processed_sharegpt/add/train_shuffle_oracle.parquet",
        llm=llm,
        model_sbert=model_sbert,
        tokenizer=tokenizer,
        tau=0.1,
    )
