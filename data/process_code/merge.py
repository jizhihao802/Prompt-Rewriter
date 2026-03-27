#用于合并不同大小的数据集
import pandas as pd

# ======================
# 输入文件
# ======================
file1 = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/test_oracle.parquet"
file2 = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/train_oracle.parquet"

# ======================
# 读取 parquet
# ======================
df1 = pd.read_parquet(file1)
df2 = pd.read_parquet(file2)

print("file1 数据量:", len(df1))
print("file2 数据量:", len(df2))

# ======================
# 合并数据
# ======================
df = pd.concat([df1, df2], ignore_index=True)

# ======================
# 随机打乱顺序
# ======================
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("合并后数据量:", len(df))

# ======================
# 保存新 parquet
# ======================
output_file = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/merged_oracle.parquet"
df.to_parquet(output_file)

print("保存完成:", output_file)