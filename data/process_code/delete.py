#删除一个数据集中存在于另一个数据集中的数据
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".jsonl", ".json"]:
        return pd.read_json(path, lines=True)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def write_table(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in [".jsonl", ".json"]:
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"不支持的输出格式: {ext}")


def diff_by_keys(source_df: pd.DataFrame, sampled_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    # 只保留采样文件中用于匹配的列，并去重
    sampled_keys = sampled_df[key_cols].drop_duplicates()
    merged = source_df.merge(sampled_keys, on=key_cols, how="left", indicator=True)
    result = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return result


def diff_by_full_row(source_df: pd.DataFrame, sampled_df: pd.DataFrame) -> pd.DataFrame:
    # 按整行比较（列名和列顺序需一致）
    if list(source_df.columns) != list(sampled_df.columns):
        raise ValueError(
            "未提供 key 列时，源文件与采样文件的列必须完全一致（列名和顺序一致）。"
        )

    def normalize_value(v):
        # 处理 numpy 标量
        if isinstance(v, np.generic):
            v = v.item()

        # 处理 None/NaN
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return "__NaN__"

        # 处理时间类型
        if isinstance(v, (pd.Timestamp, pd.Timedelta)):
            return v.isoformat()

        # 递归处理常见容器/数组
        if isinstance(v, np.ndarray):
            return [normalize_value(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [normalize_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): normalize_value(v[k]) for k in sorted(v.keys(), key=lambda x: str(x))}
        if isinstance(v, set):
            return sorted(normalize_value(x) for x in v)

        return v

    def row_signature(row_tuple):
        norm = [normalize_value(v) for v in row_tuple]
        return json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    source_sigs = [row_signature(r) for r in source_df.itertuples(index=False, name=None)]
    sampled_sigs = [row_signature(r) for r in sampled_df.itertuples(index=False, name=None)]

    # 用多重集合做差，避免重复行被错误全部剔除
    sampled_counter = Counter(sampled_sigs)
    keep_mask = []
    for sig in source_sigs:
        if sampled_counter[sig] > 0:
            sampled_counter[sig] -= 1
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    return source_df[pd.Series(keep_mask, index=source_df.index)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="源文件路径")
    parser.add_argument("--sampled", required=True, help="采样后文件路径")
    parser.add_argument("--out", required=True, help="输出新文件路径")
    parser.add_argument(
        "--keys",
        default="",
        help="用于判断是否相同的列，逗号分隔，例如 id 或 id,text；不填则按整行比较",
    )
    args = parser.parse_args()

    source_df = read_table(args.source)
    sampled_df = read_table(args.sampled)

    if args.keys.strip():
        key_cols = [k.strip() for k in args.keys.split(",") if k.strip()]
        miss = [c for c in key_cols if c not in source_df.columns or c not in sampled_df.columns]
        if miss:
            raise ValueError(f"以下 key 列在文件中不存在: {miss}")
        result_df = diff_by_keys(source_df, sampled_df, key_cols)
    else:
        result_df = diff_by_full_row(source_df, sampled_df)

    write_table(result_df, args.out)
    print(f"源文件条数: {len(source_df)}")
    print(f"采样文件条数: {len(sampled_df)}")
    print(f"差集条数(输出): {len(result_df)}")
    print(f"已保存到: {args.out}")


if __name__ == "__main__":
    main()