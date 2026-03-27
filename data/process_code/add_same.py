from datasets import Dataset, load_dataset, concatenate_datasets

# 文件路径
src_file = "/root/autodl-tmp/data/processed_sharegpt/test2.parquet"  # 来源 parquet
out_file = "/root/autodl-tmp/data/processed_sharegpt/test2_same.parquet"  # 生成的新文件

# 加载数据集
src_ds = load_dataset("parquet", data_files=src_file, split="train")

# 取第一条数据
one_item = src_ds.select(range(1))

# 复制 256 次
copied_items = [one_item] * 256  # 创建 256 个副本

# 拼接成一个新的数据集
new_ds = concatenate_datasets(copied_items)

# 写入到新文件
new_ds.to_parquet(out_file)

print(len(new_ds))
print("保存完成:", out_file)
