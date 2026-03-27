#打乱原有数据中的方法顺序
import argparse
import pandas as pd
import random
import re
from typing import Dict

# 原始语义（第1/2/3项会被随机分配）
SEMANTICS = [
    "调整指令结构",
    "缩短指令长度",
    "增加任务说明"
]
FIXED_FOUR = "无需进行优化"

# 捕获整个 <method>...</method>
METHOD_TAG_RE = re.compile(r"<method>.*?</method>", flags=re.DOTALL)


def randomize_semantics_keep_numbers() -> Dict[str, str]:
    """保持编号，但打乱前三个语义。"""
    shuffled = SEMANTICS.copy()
    random.shuffle(shuffled)
    return {
        "1": shuffled[0],
        "2": shuffled[1],
        "3": shuffled[2],
        "4": FIXED_FOUR,
    }


def build_method_tag(mapping: Dict[str, str]) -> str:
    """构造 <method>1.xxx 2.xxx 3.xxx 4.xxx</method>"""
    parts = []
    for i in ["1", "2", "3", "4"]:
        parts.append(f"{i}.{mapping[i]}")
    return "<method>" + " ".join(parts) + "</method>"


def replace_second_method_tag(content: str, new_method_tag: str) -> str:
    """
    只替换 content 中第二个 <method>...</method>，若不存在或只有一个，就原样返回。
    """
    matches = list(METHOD_TAG_RE.finditer(content))
    if len(matches) < 2:
        # 少于两个，不替换
        return content

    # 获取第二个匹配的位置
    second_match = matches[1]
    start, end = second_match.span()
    # 构造新字符串：前 + 新 tag + 后
    new_content = content[:start] + new_method_tag + content[end:]
    return new_content



def process_row(row: dict):
    """处理单条数据，替换 method 区块，并记录 mapping。"""
    prompt_list = row.get("prompt", [])
    if not prompt_list:
        return row

    content = prompt_list[0].get("content", "")
    if not isinstance(content, str):
        return row

    # ---- 随机映射 ----
    mapping = randomize_semantics_keep_numbers()

    # ---- 构造新 method tag ----
    new_method_tag = build_method_tag(mapping)

    # ---- 替换原 method 位置 ----
    new_content = replace_second_method_tag(content, new_method_tag)

    print(new_content)

    # ---- 写回 ----
    row["prompt"][0]["content"] = new_content

    # ---- 存储映射到 extra_info ----
    extra = row.get("extra_info", {})
    extra["method_mapping"] = mapping  # 例： {"1": "缩短指令长度", ...}
    print(mapping)
    row["extra_info"] = extra

    return row


def main(input_path, output_path, seed=None):
    if seed is not None:
        random.seed(seed)

    print(f"Loading parquet: {input_path}")
    df = pd.read_parquet(input_path)

    new_rows = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        processed = process_row(row_dict)
        new_rows.append(processed)

    # 保存
    out_df = pd.DataFrame(new_rows)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.input, args.output, args.seed)
