import pandas as pd

# 输入/输出文件路径
input_path = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/train_oracle.parquet"
output_path = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/sample_1000.parquet"

# 读取 parquet
df = pd.read_parquet(input_path)

# 抽样数量（如果不足1000条，就全量保存）
n = min(1000, len(df))

# 随机抽样并保存
sampled_df = df.sample(n=n, random_state=42)
sampled_df.to_parquet(output_path, index=False)

print(f"原始数据量: {len(df)}")
print(f"抽样数据量: {len(sampled_df)}")
print(f"已保存到: {output_path}")