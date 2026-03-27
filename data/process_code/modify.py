#修改已有的parquet数据文件，编辑方法选择prompt的前后缀
import argparse
import os
from datasets import Dataset
import re

def replace_prefix_suffix(text, old_prefix, old_suffix, new_prefix, new_suffix):
    """
    用于将 prompt 的前后缀替换为新的版本
    """

    # 去除旧前缀
    if old_prefix and text.startswith(old_prefix):
        text = text[len(old_prefix):]

    # 去除旧后缀
    if old_suffix and text.endswith(old_suffix):
        text = text[:-len(old_suffix)]

    # 添加新前后缀
    new_text = f"{new_prefix}{text}{new_suffix}"
    return new_text


def process_parquet(input_path, output_path, old_prefix, old_suffix, new_prefix, new_suffix):
    print(f"🔹 Loading parquet: {input_path}")
    dataset = Dataset.from_parquet(input_path)

    def modify_record(record):
        prompt_list = record["prompt"]
        if not prompt_list:
            return record

        content = prompt_list[0]["content"]

        # 替换前后缀
        new_content = replace_prefix_suffix(
            content,
            old_prefix=old_prefix,
            old_suffix=old_suffix,
            new_prefix=new_prefix,
            new_suffix=new_suffix
        )

        record["prompt"][0]["content"] = new_content
        return record

    print("🔧 Updating records...")
    new_dataset = dataset.map(modify_record)

    # 预览前几条，便于观察修改效果
    preview_n = min(args.preview_n, len(dataset))
    if preview_n > 0:
        print(f"\n👀 Preview first {preview_n} samples:")
        for i in range(preview_n):
            old_content = dataset[i]["prompt"][0]["content"] if dataset[i]["prompt"] else ""
            new_content = new_dataset[i]["prompt"][0]["content"] if new_dataset[i]["prompt"] else ""
            print("-" * 80)
            print(f"[Sample {i}] OLD:")
            print(old_content)
            print(f"[Sample {i}] NEW:")
            print(new_content)

    print(f"💾 Saving updated dataset to {output_path}")
    new_dataset.to_parquet(output_path)
    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/autodl-tmp/data/processed_sharegpt/fixed_8/test.parquet")
    parser.add_argument("--output", default="/root/autodl-tmp/data/processed_sharegpt/fixed_8/test3.parquet")

    #parser.add_argument("--old_prefix", default="你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令从<method>和</method>之间的优化方法中选择合适的一个或几个优化方式。<instruction>")
    #parser.add_argument("--old_suffix", default="</instruction>。<method>1.调整指令结构 2.缩短指令长度 3.增加任务说明 4.无需进行优化</method>只输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，不输出其他任何内容。")
    parser.add_argument("--old_prefix", default="你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令从<method>和</method>之间的优化方法组合中选择合适的一个选项。<instruction>")
    parser.add_argument("--old_suffix", default="</instruction>。<method>可选优化组合只有以下8种:"
        "1、2、3、4、2/3、2/4、3/4、2/3/4。"
        "其中:1=无需进行优化,2=缩短指令长度,3=增加任务说明,4=调整指令结构。</method>"
        "请只输出上述8种组合中的一种,不输出其他任何内容。")
    parser.add_argument("--new_prefix", default="你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令从以下 8 个选项中选择一个最合适的优化方法组合:\n可选项:1、2、3、4、2/3、2/4、3/4\n定义:1 = 无需进行优化,2 = 缩短指令长度,3 = 增加任务说明,4 = 调整指令结构\n要求:- 只能输出上述 8 个选项中的一个\n- 不要输出解释\n- 不要输出多余内容\n- 输出必须完全匹配\n<instruction>")
    parser.add_argument("--new_suffix", default="</instruction>")

    parser.add_argument("--preview_n", type=int, default=3)

    args = parser.parse_args()

    process_parquet(
        args.input,
        args.output,
        args.old_prefix,
        args.old_suffix,
        args.new_prefix,
        args.new_suffix
    )
