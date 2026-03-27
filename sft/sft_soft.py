import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

# =============================
# 加载 parquet 数据
# =============================
parquet_file = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/sample_1000.parquet"
agent_model_path = "/root/autodl-tmp/model/qwen2.5-0.5b"
tokenizer = AutoTokenizer.from_pretrained(
        agent_model_path,
        trust_remote_code=True
)
df = pd.read_parquet(parquet_file)
print("raw rows:", len(df))

# =============================
# 工具函数：从 oracle_method_scores 提取 soft 分布
# =============================
def build_soft_candidates(oracle_scores: dict, top_k: int = 3):
    """
    返回: [(completion_str, prob), ...]
    仅保留 sim_score 最高的 top_k 个组合，并在其上归一化概率
    """
    items = []
    for _, v in oracle_scores.items():
        methods = v.get("methods", None)
        if methods is None:
            continue
        completion = "/".join(methods)

        # 兼容不同分数字段
        s = v.get("sim_score", 0.0)
        items.append((completion, float(s)))

    if len(items) == 0:
        return []

    # 只保留 sim_score 排名前 top_k 的组合
    items = sorted(items, key=lambda x: x[1], reverse=True)[: max(1, int(top_k))]

    scores = np.array([x[1] for x in items], dtype=np.float64)

    # 若看起来不像概率分布，则 softmax 归一化
    # （也兼容原始 logits / 任意实数分数）
    if (scores < 0).any() or abs(scores.sum() - 1.0) > 1e-3:
        scores = np.exp(scores - scores.max())
        probs = scores / (scores.sum() + 1e-12)
    else:
        probs = scores / (scores.sum() + 1e-12)

    return [(items[i][0], float(probs[i])) for i in range(len(items))]

# =============================
# 构建 soft SFT 训练集
# =============================
train_data = []
for _, row in df.iterrows():
    instruction = row["prompt"][0]["content"]
    extra_info = row["extra_info"]
    oracle_scores = extra_info["oracle_method_scores"]

    messages = [
		{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    candidates = build_soft_candidates(oracle_scores, top_k=3)

    # fallback：没有 soft 候选时回退 hard label
    if len(candidates) == 0:
        hard_label = extra_info["oracle_hard_label"]
        best_methods = oracle_scores[hard_label]["methods"]
        candidates = [("/".join(best_methods), 1.0)]

    for completion, p in candidates:
        train_data.append(
            {
                "prompt": text,
                "completion": completion,
                "sample_weight": p,  # soft 权重
            }
        )

dataset = Dataset.from_pandas(pd.DataFrame(train_data))
print("expanded rows:", len(dataset))

# =============================
# 训练前统计（新增）
# =============================
stats_df = pd.DataFrame(train_data)

print("\n===== Pre-train Statistics =====")
print(f"expanded_samples: {len(stats_df)}")
print(f"unique_prompts: {stats_df['prompt'].nunique()}")

# completion 质量
empty_completion = (stats_df["completion"].astype(str).str.strip() == "").sum()
print(f"empty_completion: {empty_completion} ({empty_completion / max(len(stats_df),1):.2%})")

# sample_weight 质量
w = pd.to_numeric(stats_df["sample_weight"], errors="coerce")
print(f"weight_nan: {w.isna().sum()}, weight_inf: {np.isinf(w.fillna(0)).sum()}")
print(
    "weight_summary:",
    {
        "min": float(np.nanmin(w.values)),
        "max": float(np.nanmax(w.values)),
        "mean": float(np.nanmean(w.values)),
        "sum": float(np.nansum(w.values)),
    },
)

# 每个 prompt 的权重和（理想接近 1）
wsum_by_prompt = stats_df.groupby("prompt", dropna=False)["sample_weight"].sum()
print(
    "weight_sum_per_prompt:",
    {
        "min": float(wsum_by_prompt.min()),
        "p50": float(wsum_by_prompt.median()),
        "p95": float(wsum_by_prompt.quantile(0.95)),
        "max": float(wsum_by_prompt.max()),
    },
)

# completion token 长度统计（用于排查监督 token 过少）
tok = AutoTokenizer.from_pretrained(agent_model_path, trust_remote_code=True)
comp_lens = stats_df["completion"].astype(str).apply(
    lambda x: len(tok(x, add_special_tokens=False)["input_ids"])
)
print(
    "completion_token_len:",
    {
        "min": int(comp_lens.min()),
        "p50": float(comp_lens.quantile(0.5)),
        "p95": float(comp_lens.quantile(0.95)),
        "max": int(comp_lens.max()),
        "zero_len_count": int((comp_lens == 0).sum()),
    },
)
print("===== End Statistics =====\n")

# =============================
# 自定义 Trainer：加权序列损失
# =============================
class WeightedSoftSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits                  # [B, L, V]
        labels = inputs["labels"]                # [B, L]

        # 关键：Causal LM 需要 shift
        shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()       # [B, L-1]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size(0), shift_labels.size(1))

        valid_mask = (shift_labels != -100).float()
        valid_tokens_per_sample = valid_mask.sum(dim=1)

        seq_loss = (token_loss * valid_mask).sum(dim=1) / valid_tokens_per_sample.clamp_min(1.0)

        if sample_weight is None:
            loss = seq_loss.mean()
        else:
            w = sample_weight.to(seq_loss.device).float()
            loss = (seq_loss * w).sum() / w.sum().clamp_min(1e-12)

        return (loss, outputs) if return_outputs else loss

# =============================
# 模型路径 & 训练参数
# =============================

training_args = SFTConfig(
    output_dir="/root/autodl-tmp/model/qwen2.5-0.5b-soft-sft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=200,
    packing=False,  # soft 权重场景建议关闭
    remove_unused_columns=False,  # 保留 sample_weight
)

trainer = WeightedSoftSFTTrainer(
    model=agent_model_path,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()