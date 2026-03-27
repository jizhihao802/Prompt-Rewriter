import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# =============================
# 加载 parquet 数据
# =============================
parquet_file = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/test_oracle.parquet"
df = pd.read_parquet(parquet_file)
print(len(df))
# =============================
# 构建训练集
# =============================
train_data = []
for idx, row in df.iterrows():
    #提取user指令
    instruction = row["prompt"][0]["content"]
    #提取最优方法
    extra_info = row["extra_info"]

    hard_label = extra_info["oracle_hard_label"]
    oracle_scores = extra_info["oracle_method_scores"]
    best_methods = oracle_scores[hard_label]["methods"]

    method_str = "/".join(best_methods)

    full_instruction = (
        "system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
        "user\n"
        f"{instruction}\n"
        "assistant\n"
    )

    train_data.append({
        "prompt": full_instruction,
        "completion": method_str  # 输出方法 ID
    })

dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# =============================
# 加载模型和 tokenizer
# =============================
agent_model_path = "/root/autodl-tmp/model/qwen2.5-0.5b"
# =============================
# SFT 训练参数
# =============================
training_args = SFTConfig(
    output_dir="/root/autodl-tmp/model/qwen2.5-0.5b-sft_fixed8",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=15,
    logging_steps=10,
    save_steps=50,
    packing=True,  # 启用示例打包，提高训练效率
)

# =============================
# 创建 SFTTrainer
# =============================
trainer = SFTTrainer(
    model=agent_model_path,
    train_dataset=dataset,
    args=training_args,
)

# =============================
# 开始微调
# =============================
trainer.train()
