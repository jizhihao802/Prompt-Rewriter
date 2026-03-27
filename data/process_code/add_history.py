#处理filter2.py处理后的数据，将每组对话作为独立的一个sample，并加入对话历史
import json
import argparse
def load_json_or_jsonl(file_path):
    """兼容普通 JSON 和 JSON Lines (NDJSON) 格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]
        
def convert_one_sample(sample):
    """
    输入数据格式：
    {
        "id": "",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ],
        "lang": ""
    }
    """

    conv = sample["conversations"]
    lang = sample.get("lang", "")
    sid = sample.get("id", "")

    results = []

    # 必须保证 human/gpt 成对
    if len(conv) % 2 != 0:
        return results

    # 遍历每一轮对话
    for i in range(0, len(conv), 2):
        human_msg = conv[i]
        gpt_msg = conv[i + 1]

        # 只允许 human -> gpt
        if human_msg["from"] != "human" or gpt_msg["from"] != "gpt":
            continue

        # 构建历史：之前的所有对话
        history_items = conv[:i]
        history_text = ""

        for h in range(0, len(history_items), 2):
            h_user = history_items[h]
            h_ass = history_items[h + 1]

            history_text += f"<user>: {h_user['value']}\n<assistant>: {h_ass['value']}\n"

        # 去掉最后多余的换行
        history_text = history_text.strip()

        # 构建输出样本
        out = {
            "id": f"{sid}_round_{i//2 + 1}",
            "history": history_text,
            "instruction": human_msg["value"],
            "response": gpt_msg["value"],
            "lang": lang
        }

        results.append(out)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/Users/jizhihaosmac/Documents/sharegpt_zh_38K_filtered_3rounds_200_450_no_url.json")
    parser.add_argument("--output_path", default="/Users/jizhihaosmac/Documents/sharegpt_zh_38K_filtered_3rounds_200_450_no_url_history.json")
    args = parser.parse_args()
   
    input_path = args.input_path
    output_path = args.output_path

    data = load_json_or_jsonl(input_path)

    all_samples = []

    for sample in data:
        all_samples.extend(convert_one_sample(sample))

    print(len(all_samples))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)