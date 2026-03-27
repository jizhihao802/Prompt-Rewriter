#用来筛选指定长度范围和指定轮数的对话数据，为加入对话历史准备
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
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def contains_url(text):
    """检查文本中是否包含 URL"""
    url_pattern = re.compile(r"(https?://\S+)|(www\.\S+)", re.IGNORECASE)
    return bool(url_pattern.search(text))

def filter_json(
    input_path, output_path,
    min_human_len=50, min_gpt_len=75,
    max_human_len=700, max_gpt_len=700,
    max_rounds=1
):
    data = load_json_or_jsonl(input_path)
    filtered = []

    for sample in data:
        conv = sample.get("conversations", [])

        # 必须至少包含 6 条消息（3 组 human→gpt）
        if len(conv) < 2*max_rounds:
            continue

        valid = True
        selected_pairs = []

        # 只检查前三组对话
        for k in range(max_rounds):
            h = conv[2*k]
            g = conv[2*k + 1]

            # 必须是 human → gpt
            if h.get("from") not in ["human", "user"] or g.get("from") not in ["gpt", "assistant"]:
                valid = False
                break

            human_text = h.get("value", "").strip()
            gpt_text = g.get("value", "").strip()

            # human 中不能包含 URL（所有 3 组都检查）
            if contains_url(human_text):
                valid = False
                break

            # ===== 仅第 1 组检查长度 =====
            if k == 0:
                human_len = count_length(human_text)
                gpt_len = count_length(gpt_text)

                if not (min_human_len <= human_len <= max_human_len):
                    valid = False
                    break
                if not (min_gpt_len <= gpt_len <= max_gpt_len):
                    valid = False
                    break

            selected_pairs.append(h)
            selected_pairs.append(g)

        # 三组全部合法才能加入
        if valid and len(selected_pairs) == 2*max_rounds:
            filtered.append({
                "id": sample.get("id", ""),
                "conversations": selected_pairs,
                "lang": sample.get("lang", "")
            })

    # 写出结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"✅ 筛选完成：共保留 {len(filtered)} 条样本。")
    print(f"📁 输出文件：{output_path}")



# 示例调用
filter_json(
    "/Users/jizhihaosmac/Documents/sharegpt_zh_38K_format.jsonl",
    "/Users/jizhihaosmac/Documents/sharegpt_zh_38K_filtered_3rounds_200_450_no_url.json",
    min_human_len=200,
    min_gpt_len=200,
    max_human_len=450,
    max_gpt_len=450,
    max_rounds=3
)
