import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

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
# 读取数据
# =============================

parquet_file = "/root/autodl-tmp/data/processed_sharegpt/add/train_shuffle_oracle.parquet"

df = pd.read_parquet(parquet_file)

print("数据量:", len(df))


# =============================
# 构建训练数据
# =============================

train_data = []

for _, row in df.iterrows():

    #instruction = row["prompt"][0]["content"]

    extra_info = row["extra_info"]

    instruction = extra_info["instruction"]

    hard_label = extra_info["oracle_hard_label"]

    oracle_scores = extra_info["oracle_method_scores"]

    methods = oracle_scores[hard_label]["methods"]

    # 关键：编号 -> 文本
    method_text_map = extra_info.get("method_mapping", {})

    selected_texts = [method_text_map.get(m, "") for m in methods]

    selected_texts = [t for t in selected_texts if t]

    label = methods_text_to_label(selected_texts)

    train_data.append({
        "text": instruction,
        "label": label
    })


print("训练样本:", len(train_data))


dataset = Dataset.from_pandas(pd.DataFrame(train_data))


# =============================
# 加载模型
# =============================

model_path = "/root/autodl-tmp/model/qwen2.5-0.5b"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=8,
    trust_remote_code=True
)

print(model.dtype)
tokenizer.save_pretrained("/root/autodl-tmp/model/method_classifier/c4/tokenizer")

#print(tokenizer.pad_token)
#print(tokenizer.pad_token_id)
#print(model.config.pad_token_id)

model.config.pad_token_id = tokenizer.pad_token_id

#print(tokenizer.pad_token)
#print(tokenizer.pad_token_id)
#print(model.config.pad_token_id)

# =============================
# tokenize
# =============================

def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )


dataset = dataset.map(tokenize, batched=True)


# =============================
# 训练参数
# =============================

training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/model/method_classifier/c4",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=15,
    logging_steps=10,
    save_steps=100,
    save_total_limit=30,
    bf16=True,
    seed=42
)


# =============================
# trainer
# =============================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


trainer.train()
