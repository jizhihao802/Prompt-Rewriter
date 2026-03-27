#筛选指定长度范围的对话数据，并且只保留第一组对话
import json
import re

def count_length(text):
    """判断中英文混合文本的长度（中文按字数，英文按单词数）"""
    text = text.strip()
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return len(text)
    else:
        return len(text.split())

def load_json_or_jsonl(file_path):
    """兼容普通 JSON 和 JSON Lines (NDJSON) 格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)  # 标准 JSON 数组
        else:
            return [json.loads(line) for line in f if line.strip()]  # 按行解析 JSONL

def contains_url(text):
    """检查文本中是否包含 URL"""
    url_pattern = re.compile(
        r"(https?://\S+)|(www\.\S+)", re.IGNORECASE
    )
    return bool(url_pattern.search(text))

def filter_json(input_path, output_path, 
                min_human_len=50, min_gpt_len=75, 
                max_human_len=700, max_gpt_len=700):
    """筛选满足最小和最大长度要求的对话，同时去掉含 URL 的 human prompt"""
    data = load_json_or_jsonl(input_path)
    filtered = []

    for sample in data:
        conv = sample.get("conversations", [])
        if len(conv) >= 2 and conv[0].get("from") in ["human", "user"] and conv[1].get("from") in ["gpt", "assistant"]:
            human_text = conv[0].get("value", "").strip()
            gpt_text = conv[1].get("value", "").strip()

            # ===== 新增：去掉包含 URL 的 human prompt =====
            if contains_url(human_text):
                continue

            human_len = count_length(human_text)
            gpt_len = count_length(gpt_text)

            if (min_human_len <= human_len <= max_human_len) and (min_gpt_len <= gpt_len <= max_gpt_len):
                new_entry = {
                    "id": sample.get("id", ""),
                    "conversations": [conv[0], conv[1]],
                    "lang": sample.get("lang", "")
                }
                filtered.append(new_entry)

    # 写出筛选后的结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"✅ 筛选完成，共保留 {len(filtered)} 条样本（human: {min_human_len}-{max_human_len}, gpt: {min_gpt_len}-{max_gpt_len}）。")
    print(f"📁 输出文件：{output_path}")

# 示例调用
filter_json(
    "/root/autodl-tmp/data/sharegpt-chinese-english/sharegpt_jsonl/common_zh_70k_sharegpt.jsonl",
    "/root/autodl-tmp/data/sharegpt-chinese-english/sharegpt_jsonl/common_zh_70k_sharegpt_filtered_200_500_no_url.json",
    min_human_len=200,
    min_gpt_len=200,
    max_human_len=500,
    max_gpt_len=500
)
