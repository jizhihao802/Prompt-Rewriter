import pandas as pd
import numpy as np

# ======================
# parquet路径
# ======================
#parquet_file = "/root/autodl-tmp/data/processed_sharegpt/train/train_oracle.parquet"
parquet_file = "/root/autodl-tmp/data/processed_sharegpt/fixed_8/train_oracle.parquet"
# ======================
# 读取数据
# ======================
df = pd.read_parquet(parquet_file)

print("数据量:", len(df))

# ======================
# 存储每个排名位置的sim_score
# ======================
rank_scores = {}

# 存储所有sim_score（用于random baseline）
all_scores = []

# ======================
# 遍历数据
# ======================
for _, row in df.iterrows():

    extra_info = row["extra_info"]
    oracle_scores = extra_info["oracle_method_scores"]

    # 取出所有 sim_score
    scores = []

    for k in oracle_scores:
        s = oracle_scores[k]["sim_score"]
        scores.append(s)
        all_scores.append(s)

    # 从高到低排序
    scores = sorted(scores, reverse=True)

    # 按排名存储
    for rank, score in enumerate(scores):

        if rank not in rank_scores:
            rank_scores[rank] = []

        rank_scores[rank].append(score)

# ======================
# 计算 Rank mean
# ======================
print("\n=== Rank mean sim_score ===\n")

rank_means = {}

for rank in sorted(rank_scores.keys()):

    mean_score = np.mean(rank_scores[rank])
    rank_means[rank] = mean_score

    print(f"Rank {rank+1}: {mean_score:.4f}")

# ======================
# Oracle performance
# ======================
oracle_performance = rank_means[0]

print("\nOracle performance:", round(oracle_performance, 4))

# ======================
# Random performance
# ======================
random_performance = np.mean(all_scores)

print("Random performance:", round(random_performance, 4))

# ======================
# Top-k gap
# ======================
print("\n=== Top-k gap ===\n")

for rank in sorted(rank_means.keys())[1:]:

    gap = rank_means[0] - rank_means[rank]

    print(f"Top-{rank+1} gap: {gap:.4f}")