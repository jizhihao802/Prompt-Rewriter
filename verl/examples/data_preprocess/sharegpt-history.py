#进行加入对话历史的sharegpt数据预处理
import argparse
import os
import json
import torch
import re
from datasets import Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from verl.utils.hdfs_io import copy, makedirs
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_json_or_jsonl(file_path):
    """兼容普通 JSON 和 JSONL 格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


# ======================
#   vLLM 版生成函数
# ======================
def generate_local_response(prompt: str, llm, sampling_params, max_new_tokens=0):
    # === 处理 max_new_tokens ===
    if max_new_tokens == 0:
        max_new_tokens = len(prompt)  # 保持原逻辑

    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens
    )

    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()
    return text


def extract_instruction(text: str) -> str:
    match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def compute_similarity(sbert, text1, text2):
    emb1 = sbert.encode(text1, convert_to_tensor=True)
    emb2 = sbert.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def convert_to_record(entry, split, idx):

    history = entry.get("history", "").strip()
    human = entry.get("instruction", "").strip()
    gpt = entry.get("response", "").strip()

    if not human or not gpt:
        return None

    # 计算 ground truth token 数
    tokens = tokenizer.encode(gpt, add_special_tokens=False)
    num_tokens_truth = len(tokens)
    tokens = tokenizer.encode(human, add_special_tokens=False)
    num_tokens_instruction = len(tokens)

    # ======== 构造用于选择优化方法的 prompt（保持原逻辑）=======
    question = (
        args.prefix_instruction.strip()
        + history
        + args.middle_instruction.strip()
        + human
        + args.suffix_instruction.strip()
    )

    # ======================================================
    #   1⃣️ 生成 原始回答（不使用方法提示）
    #      prompt = history + human
    # ======================================================
    if history:
        orig_prompt = f"{history}\n<user>: {human}\n<assistant>:"
    else:
        orig_prompt = f"<user>: {human}\n<assistant>:"

    try:
        orig_response = generate_local_response(
            orig_prompt, llm, sampling_params, num_tokens_truth
        )
    except Exception as e:
        print(f"⚠️ Orig generation failed at {idx}: {e}")
        orig_response = ""

    # similarity 原始回答 vs ground truth
    try:
        orig_sim = compute_similarity(sbert, orig_response, gpt)
    except:
        orig_sim = 0.0

    # ======================================================
    #   2⃣️ Baseline 模式：直接改写指令（包含历史）
    # ======================================================
    baseline_rewrite_prompt = (
        "你是一个指令优化器，负责优化用户给到大模型的指令，请结合<history>和</history>之间的对话历史在保留信息完整的前提下优化<instruction>和</instruction>之间的指令。"
        f"<history>\n{history}\n</history>\n"
        f"<instruction>\n{human}\n</instruction>"
        "并按照“<result>你优化后的指令</result>”的格式输出结果，并在</result>后终止输出。"
    )

    try:
        baseline_instruction = generate_local_response(baseline_rewrite_prompt, llm, sampling_params, num_tokens_instruction)
    except:
        baseline_instruction = human

    baseline_instruction = extract_instruction(baseline_instruction)

    # ======================================================
    #   3⃣️ Baseline 回答
    #      prompt = history + baseline_instruction
    # ======================================================
    if history:
        baseline_prompt = f"{history}\n<user>: {baseline_instruction}\n<assistant>:"
    else:
        baseline_prompt = f"<user>: {baseline_instruction}\n<assistant>:"

    try:
        baseline_response = generate_local_response(
            baseline_prompt, llm, sampling_params, num_tokens_truth
        )
    except:
        baseline_response = ""

    try:
        baseline_sim = compute_similarity(sbert, baseline_response, gpt)
    except:
        baseline_sim = 0.0

    # ======================================================
    #   生成最终 RL 训练格式
    # ======================================================

    record = {
        "data_source": "sharegpt_history",
        "prompt": [{"role": "user", "content": question}],
        "ability": "general",
        "reward_model": {"style": "rule", "ground_truth": gpt},
        "extra_info": {
            "split": split,
            "instruction": human,
            "history": history,
            "orig_similarity": round(orig_sim, 4),
            "baseline_similarity": round(baseline_sim, 4),
        },
    }
    return record



def process_split(data_split, split_name):
    processed = []
    print(f"🚀 Processing {split_name} split with {len(data_split)} samples...")
    for i, entry in enumerate(tqdm(data_split, desc=f"Processing {split_name}", ncols=100)):
        rec = convert_to_record(entry, split_name, i)
        if rec:
            processed.append(rec)
    print(f"✅ Finished {split_name}: {len(processed)} valid samples")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--local_save_dir", default="/root/autodl-tmp/data/processed_sharegpt")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--split_ratio", type=float, default=0.8)

    # 模型路径
    parser.add_argument("--local_model_path", default="/root/autodl-tmp/model/qwen3-8b")
    parser.add_argument("--sbert_model_path", default="/root/autodl-tmp/model/all-mpnet-base-v2")

    # 指令模板
    parser.add_argument("--prefix_instruction", type=str, default=(
        "你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令"
        "从<method>和</method>之间的优化方法中选择合适的一个或几个优化方式。<instruction>"
    ))
    parser.add_argument("--suffix_instruction", type=str, default=(
        "</instruction>。<method>1.调整指令结构 2.缩短指令长度 3.增加任务说明 4.无需进行优化</method>"
        "只输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，不输出其他任何内容。"
    ))

    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    # ===================
    #   加载 Tokenizer
    # ===================
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)

    # ===================
    #   加载 vLLM
    # ===================
    print(f"🔹 Loading vLLM model from {args.local_model_path}")
    llm = LLM(
        model=args.local_model_path,
        tensor_parallel_size=2,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=300,
    )

    # ===================
    #   加载 SBERT
    # ===================
    sbert = SentenceTransformer(args.sbert_model_path, device="cpu")

    # ===================
    #   加载数据
    # ===================
    data = load_json_or_jsonl(args.input_path)
    print(f"✅ Loaded {len(data)} samples")

    split_index = int(len(data) * args.split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    train_processed = process_split(train_data, "train")
    test_processed = process_split(test_data, "test")

    train_dataset = Dataset.from_list(train_processed)
    test_dataset = Dataset.from_list(test_processed)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"💾 Saved to {local_save_dir}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
