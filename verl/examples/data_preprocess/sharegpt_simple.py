import argparse
import os
import json
import torch
import random
from datasets import Dataset
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

def convert_to_record(entry, split, idx):
    conv = entry.get("conversations", [])
    if len(conv) < 2:
        return None
    human = conv[0].get("value", "").strip()
    gpt = conv[1].get("value", "").strip()
    if not human or not gpt:
        return None

    # ====== 拼接选择方法 prompt ======
    question = f"{args.prefix_instruction.strip()}{human}{args.suffix_instruction.strip()}"
    
    record = {
        "data_source": "sharegpt",
        "prompt": [{"role": "user", "content": question}],
        "ability": "general",
        "reward_model": {"style": "rule", "ground_truth": gpt},
        "extra_info": {
            "split": split,
            "instruction": human,
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
    parser.add_argument("--local_save_dir", default="/root/autodl-tmp/data/processed_sharegpt/fixed_8")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--split_ratio", type=float, default=0.9)

    # 指令模板
    parser.add_argument("--prefix_instruction", type=str, default=(
        "你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令"
        "从<method>和</method>之间的优化方法组合中选择合适的一个选项。<instruction>"
    ))
    parser.add_argument("--suffix_instruction", type=str, default=(
        "</instruction>。<method>可选优化组合只有以下8种:"
        "1、2、3、4、2/3、2/4、3/4、2/3/4。"
        "其中:1=无需进行优化,2=缩短指令长度,3=增加任务说明,4=调整指令结构。</method>"
        "请只输出上述8种组合中的一种,不输出其他任何内容。"
    ))

    args = parser.parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # ===================
    #   加载数据
    # ===================
    data = load_json_or_jsonl(args.input_path)
    print(f"✅ Loaded {len(data)} samples")

    random.shuffle(data)   # ⭐ 打乱数据

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
