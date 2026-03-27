# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess ShareGPT-style dataset to parquet format, 
adding prefix/suffix instructions around each human question.
"""

"""
Preprocess ShareGPT-style dataset to parquet format,
adding prefix/suffix instructions around each human question,
and computing baseline response similarity with SBERT.
"""

import argparse
import os
import json
import torch
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from verl.utils.hdfs_io import copy, makedirs
from tqdm import tqdm

def load_json_or_jsonl(file_path):
    """兼容普通 JSON 和 JSONL 格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def generate_local_response(prompt: str, model, tokenizer, device, max_new_tokens=0):
    """
    使用本地 LM 生成文本，并只返回模型新生成的部分（去掉输入 prompt）
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    if max_new_tokens == 0:
        max_new_tokens=len(prompt)

    with torch.no_grad():
        # 生成序列
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解码整个输出（包含输入 + 新生成部分）
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ====== 去掉 prompt 对应部分，只保留新生成内容 ======
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
    if full_text.startswith(prompt_text):
        new_text = full_text[len(prompt_text):].strip()
    else:
        new_text = full_text  # 如果没有匹配，直接返回全部

    return new_text

def extract_instruction(text: str) -> str:
    """
    从模型输出中提取 <result> ... </result> 内的内容。
    若未找到，返回原文本。
    """
    match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return text.strip()

def compute_similarity(sbert, text1, text2):
    """计算 SBERT 余弦相似度"""
    emb1 = sbert.encode(text1, convert_to_tensor=True)
    emb2 = sbert.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def convert_to_record(entry, split, idx):
    conv = entry.get("conversations", [])
    if len(conv) < 2:
        return None
    human = conv[0].get("value", "").strip()
    gpt = conv[1].get("value", "").strip()
    if not human or not gpt:
        return None

    tokens = tokenizer.encode(gpt, add_special_tokens=False)  # 不计入特殊 token
    num_tokens_truth = len(tokens)

    # ====== 构造你的原“选择优化方法”版的 question ======
    question = f"{args.prefix_instruction.strip()}{human}{args.suffix_instruction.strip()}"

    # ====== 生成原始指令的回答 ======
    try:
        orig_response = generate_local_response(human, model, tokenizer, device, num_tokens_truth)
    except Exception as e:
        print(f"⚠️ Generation failed at {idx}: {e}")
        return None

    # ====== 计算原始回答的相似度 ======
    try:
        orig_sim = compute_similarity(sbert, orig_response, gpt)
    except Exception as e:
        print(f"⚠️ SBERT failed at {idx}: {e}")
        orig_sim = 0.0

    # ============================================================
    # ✅ 生成 baseline：不给方法提示，让模型自己改写原指令
    # ============================================================

    baseline_rewrite_prompt = (
        "你是一个指令优化器，请优化<instruction>和</instruction>之间的指令。"
        f"<instruction>\n{human}\n</instruction>"
        "并按照“<result>你优化后的指令</result>”的格式输出结果，并在</result>后终止输出。"
    )

    # ---- 生成 baseline_instruction ----
    try:
        baseline_instruction = generate_local_response(
            baseline_rewrite_prompt, model, tokenizer, device
        )
    except Exception as e:
        print(f"⚠️ Baseline rewrite failed at {idx}: {e}")
        baseline_instruction = human  # fallback

    baseline_instruction = extract_instruction(baseline_instruction)

    # ---- 用 baseline_instruction 获得模型新回答 ----
    try:
        baseline_response = generate_local_response(
            baseline_instruction, model, tokenizer, device, num_tokens_truth
        )
    except Exception as e:
        print(f"⚠️ Baseline response failed at {idx}: {e}")
        baseline_response = ""

    # ---- baseline 与 ground truth 的相似度 ----
    try:
        baseline_sim = compute_similarity(sbert, baseline_response, gpt)
    except Exception as e:
        print(f"⚠️ SBERT baseline failed at {idx}: {e}")
        baseline_sim = 0.0

    # ============================================================

    record = {
        "data_source": "sharegpt",
        "prompt": [{"role": "user", "content": question}],
        "ability": "general",
        "reward_model": {"style": "rule", "ground_truth": gpt},
        "extra_info": {
            "split": split,
            "instruction": human,
            # ====== 加入原始指令得到的回答 ======
            #"orig_response": orig_response,
            "orig_similarity": round(orig_sim, 4),
            # ====== 加入 baseline 信息 ======
            #"baseline_instruction": baseline_instruction,
            #"baseline_response": baseline_response,
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

    # === 模型路径参数 ===
    parser.add_argument("--local_model_path", default="/root/autodl-tmp/model/qwen3-8b", help="回答模型路径")
    parser.add_argument("--sbert_model_path", type=str, default="/root/autodl-tmp/model/all-mpnet-base-v2")

    # === 指令模板 ===
    parser.add_argument("--prefix_instruction", type=str, default=(
        "你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令"
        "从<method>和</method>之间的优化方法中选择合适的一个或几个优化方式。<instruction>"
    ))
    parser.add_argument("--suffix_instruction", type=str, default=(
        "</instruction>。<method>0.无需进行优化 1.缩短指令长度 2.调整指令结构 3.增加任务说明</method>"
        "只输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，不输出其他任何内容。"
    ))

    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # === 设备选择 ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")

    # === 加载本地回答模型 ===
    print(f"🔹 Loading local response model from {args.local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.local_model_path, torch_dtype=torch.float16).to(device)
    model.eval()

    # === 加载 SBERT ===
    print(f"🔹 Loading SBERT model from {args.sbert_model_path}")
    sbert = SentenceTransformer(args.sbert_model_path, device=device)

    # === 加载数据 ===
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

    print(f"💾 Saved {len(train_processed)} training samples and {len(test_processed)} test samples to {local_save_dir}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print(f"☁️ Copied parquet files to HDFS: {args.hdfs_dir}")



#python3 examples/data_preprocess/sharegpt.py --input_path /root/autodl-tmp/data/sharegpt-cn/sharegpt_zh_38K_filtered_200_500_no_url.json --local_save_dir /root/autodl-tmp/data/processed_sharegpt