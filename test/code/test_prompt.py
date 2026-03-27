import argparse
import re
from typing import Optional, List, Dict, Any

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_combo_key(solution_str: str) -> Optional[str]:
    """提取并归一化 8 种固定组合之一：
    1、2、3、4、2/3、2/4、3/4、2/3/4
    """
    if solution_str is None:
        return None

    cleaned = str(solution_str).strip().replace(" ", "")
    match = re.search(r"([1-4](?:/[1-4]){0,2})", cleaned)
    if not match:
        return None

    parts = [p for p in match.group(1).split("/") if p in {"1", "2", "3", "4"}]
    if not parts:
        return None

    # 1 不能与其他方法共存
    if "1" in parts and len(parts) > 1:
        normalized = "1"
    elif "1" in parts:
        normalized = "1"
    else:
        normalized = "/".join(sorted(set(parts), key=lambda x: int(x)))

    valid = {"1", "2", "3", "4", "2/3", "2/4", "3/4", "2/3/4"}
    return normalized if normalized in valid else None


def combo_to_method_texts(combo_key: Optional[str]) -> List[str]:
    method_text_map = {
        "1": "无需进行优化",
        "2": "缩短指令长度",
        "3": "增加任务说明",
        "4": "调整指令结构",
    }
    if not combo_key:
        return []
    return [method_text_map[i] for i in combo_key.split("/") if i in method_text_map]


def build_choose_prompt(instruction: str, prefix_instruction: str, suffix_instruction: str) -> str:
    return f"{prefix_instruction.strip()}{instruction}{suffix_instruction.strip()}"


def evaluate_parquet(
    parquet_path: str,
    llm_agent,
    prefix_instruction: str = "",
    suffix_instruction: str = "",
    log_path: str = "/root/autodl-tmp/test/test_prompt_log.txt",
    output_path: Optional[str] = None,
    tokenizer=None
):
    def agent_choose_methods(instruction: str) -> str:
        out = llm_agent.generate([instruction], sampling_agent)
        return out[0].outputs[0].text.strip()

    df = pd.read_parquet(parquet_path)

    sampling_agent = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=10,
        n=1,
        logprobs=0,
        ignore_eos=False,
        repetition_penalty=1.0,
    )

    results: List[Dict[str, Any]] = []

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("idx\tsolution_str\tcombo_key\tselected_texts\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            extra_info = row.get("extra_info", {})
            original_instruction = str(extra_info.get("instruction", "")).strip()
            if not original_instruction:
                continue

            choose_prompt = build_choose_prompt(
                instruction=original_instruction,
                prefix_instruction=prefix_instruction,
                suffix_instruction=suffix_instruction,
            )

            #full_instruction = (
            #    f"system\n"
            #    f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
            #    f"user\n"
            #    f"{choose_prompt}\n"
            #    f"assistant\n"
            #)

            messages = [
				{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": choose_prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                #enable_thinking=False,  # Set to False to strictly disable thinking
            )

            print(f"[Row {idx}] Generated text: {text}")
            solution_str = agent_choose_methods(text)
            combo_key = extract_combo_key(solution_str)
            selected_texts = combo_to_method_texts(combo_key)

            print(f"[Row {idx}] solution_str: {solution_str}")
            print(f"[Row {idx}] combo_key: {combo_key}")
            print(f"[Row {idx}] selected_texts: {selected_texts}")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{idx}\t{solution_str}\t{combo_key}\t{','.join(selected_texts) if selected_texts else '[]'}\n"
                )

            results.append(
                {
                    "idx": idx,
                    "instruction": original_instruction,
                    "choose_prompt": choose_prompt,
                    "raw_choice": solution_str,
                    "combo": combo_key,
                    "selected_methods": selected_texts,
                }
            )

        except Exception as e:
            print(f"[Warning] row={idx} failed: {e}")

    out_df = pd.DataFrame(results)
    if output_path:
        out_df.to_parquet(output_path, index=False)
        print(f"已保存结果到: {output_path}")

    print(f"已保存日志到: {log_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, default="/root/autodl-tmp/data/processed_sharegpt/fixed_8/test.parquet")
    parser.add_argument("--agent_model", type=str, default="/root/autodl-tmp/model/qwen2.5-0.5b")
    parser.add_argument("--log_path", type=str, default="/root/autodl-tmp/test/test_prompt/log.txt")
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument(
        "--prefix_instruction",
        type=str,
        default=(
            "你是一个指令优化器，请根据<instruction>和</instruction>之间待优化的指令从<method>和</method>之间的优化方法组合中选择合适的一个选项。<instruction>"
        ),
    )
    parser.add_argument(
        "--suffix_instruction",
        type=str,
        default=(
            "</instruction>。<method>可选优化组合只有以下8种:"
        "1、2、3、4、2/3、2/4、3/4、2/3/4。"
        "其中:1=无需进行优化,2=缩短指令长度,3=增加任务说明,4=调整指令结构。</method>"
        "请只输出上述8种组合中的一种,不输出其他任何内容。"
        ),
    )

    parser.add_argument("--agent_tensor_parallel", type=int, default=2)
    parser.add_argument("--agent_gpu_memory_utilization", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/model/qwen2.5-0.5b",
        trust_remote_code=True
    )

    llm_agent = LLM(
        model=args.agent_model,
        tensor_parallel_size=args.agent_tensor_parallel,
        trust_remote_code=True,
        gpu_memory_utilization=args.agent_gpu_memory_utilization,
        skip_tokenizer_init=False,
        disable_log_stats=True,
        seed=42,
    )

    evaluate_parquet(
        parquet_path=args.parquet_path,
        llm_agent=llm_agent,
        prefix_instruction=args.prefix_instruction,
        suffix_instruction=args.suffix_instruction,
        log_path=args.log_path,
        output_path=args.output_path,
        tokenizer=tokenizer,
    )
