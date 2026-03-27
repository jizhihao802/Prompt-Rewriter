from datasets import Dataset, load_dataset, concatenate_datasets

src_file = "/root/autodl-tmp/data/processed_sharegpt/test2.parquet"     # 来源 parquet
dst_file = "/root/autodl-tmp/data/processed_sharegpt/train2.parquet"     # 目标 parquet
out_file = "/root/autodl-tmp/data/processed_sharegpt/train2_1024.parquet" # 生成的新文件

# 加载数据集
src_ds = load_dataset("parquet", data_files=src_file, split="train")
dst_ds = load_dataset("parquet", data_files=dst_file, split="train")

# 取一条数据
one_item = src_ds.select(range(1))

# 拼接（使用 concatenate_datasets）
new_ds = concatenate_datasets([dst_ds, one_item])

# 写入到新文件
new_ds.to_parquet(out_file)

print(len(new_ds))
print("保存完成:", out_file)
