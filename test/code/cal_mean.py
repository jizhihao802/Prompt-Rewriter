#统计RL训练日志中的critic/rewards/mean的值，计算每20条的均值，并画出折线图
import re
import statistics
import matplotlib.pyplot as plt

def group_means_from_log(file_path: str, key: str = "critic/rewards/mean", total: int = 520, group_size: int = 20):
    # 匹配例如：critic/rewards/mean:0.7078563570976257
    pattern = re.compile(
        rf"{re.escape(key)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    values = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))

    if len(values) < total:
        raise ValueError(f"只找到 {len(values)} 条 {key}，少于 {total} 条。")

    # 如果超过 520 条，只取前 520 条
    values = values[:total]

    if total % group_size != 0:
        raise ValueError("total 必须能被 group_size 整除。")

    means = []
    for i in range(0, total, group_size):
        chunk = values[i:i + group_size]
        means.append(statistics.mean(chunk))

    return means


if __name__ == "__main__":
    log_file = "/root/autodl-tmp/verl/verl_demo18.log"  # 改成你的文件路径
    total = 520
    group_size = 20
    means_26 = group_means_from_log(log_file, total=total, group_size=group_size)

    print(f"共得到 {len(means_26)} 个均值：")
    for i, v in enumerate(means_26, 1):
        print(f"{i:02d}: {v:.6f}")

    # 横轴为步数：20, 40, ..., 520（每20条做一次均值）
    steps = list(range(group_size, total + 1, group_size))

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, means_26, marker="o", linewidth=1.8)
    plt.title("Mean of critic/rewards/mean per 20 steps")
    plt.xlabel("Step")
    plt.ylabel("Mean")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(steps)
    plt.tight_layout()

    out_png = "critic_rewards_mean_line.png"
    plt.savefig(out_png, dpi=150)
    print(f"折线图已保存到: {out_png}")

    # 本地有图形界面时可弹窗显示
    # plt.show()