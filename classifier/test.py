import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================
# 配置路径
# =============================
model_dir = "/root/autodl-tmp/model/method_classifier/c4/checkpoint-3400"
tokenizer_dir = "/root/autodl-tmp/model/method_classifier/c4/tokenizer"
test_parquet_file = "/root/autodl-tmp/data/processed_sharegpt/add/test_shuffle_oracle.parquet"

# =============================
# 方法文本 -> bit
# =============================
METHOD_BIT = {
    "调整指令结构":1,
    "缩短指令长度":2,
    "增加任务说明":4
}

# =============================
# 方法文本 -> label
# =============================
def methods_text_to_label(texts):
    label = 0
    for t in texts:
        if "无需进行优化" in t:
            return 0
        for key in METHOD_BIT:
            if key in t:
                label |= METHOD_BIT[key]
    return label

# =============================
# label -> 方法文本
# =============================

def label_to_methods(label):
    if label == 0:
        return ["无需进行优化"]

    methods = []
    for name, bit in METHOD_BIT.items():
        if label & bit:
            methods.append(name)

    return methods

def text_to_method_id(method_texts, method_text_map):
    ids = []
    for mid, text in method_text_map.items():
        for t in method_texts:
            if t in text:
                ids.append(mid)
    return sorted(ids)

# =============================
# 加载测试数据
# =============================
df = pd.read_parquet(test_parquet_file)
print("测试数据量:", len(df))

test_data = []
for _, row in df.iterrows():
    extra_info = row["extra_info"]
    instruction = extra_info["instruction"]
    hard_label = extra_info["oracle_hard_label"]
    oracle_scores = extra_info["oracle_method_scores"]
    methods = oracle_scores[hard_label]["methods"]
    method_text_map = extra_info.get("method_mapping", {})
    selected_texts = [method_text_map.get(m, "") for m in methods]
    selected_texts = [t for t in selected_texts if t]
    label = methods_text_to_label(selected_texts)
    test_data.append({"text": instruction, "label": label})

test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

# =============================
# 加载 tokenizer 和模型
# =============================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =============================
# tokenize
# =============================
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

test_dataset = test_dataset.map(tokenize, batched=True)

test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =============================
# 推理并计算准确度
# =============================
all_preds = []
all_labels = []
sim_scores = []

for idx, sample in enumerate(test_dataset):

    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device)
    }

    label = sample["label"].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

    all_preds.append(pred)
    all_labels.append(label)

    # =============================
    # 计算 sim_score
    # =============================

    row = df.iloc[idx]
    extra_info = row["extra_info"]

    oracle_scores = extra_info["oracle_method_scores"]
    method_text_map = extra_info.get("method_mapping", {})

    # 1. 预测类别 -> 方法文本
    pred_method_texts = label_to_methods(pred)

    # 2. 方法文本 -> 方法ID
    pred_method_ids = text_to_method_id(pred_method_texts, method_text_map)

    #print(pred_method_ids)

    sim_score = None

    for combo_id, info in oracle_scores.items():
        combo_methods = info.get("methods", [])
        #print(combo_methods)
        if set(combo_methods) == set(pred_method_ids):
            sim_score = float(info.get("sim_score", 0.0))
            #print(
            #    f"✅ HIT oracle combo={combo_id} "
            #    f"methods={combo_methods} "
            #    f"sim={sim_score:.4f}"
            #)
            break

    if sim_score is not None:
        sim_scores.append(sim_score)

# 分类准确率
correct = sum([p == l for p, l in zip(all_preds, all_labels)])
accuracy = correct / len(all_labels)

# sim_score 均值
avg_sim_score = sum(sim_scores) / len(sim_scores)

print(f"测试集准确度: {accuracy*100:.2f}%")
print(f"预测方法 sim_score 平均值: {avg_sim_score:.4f}")