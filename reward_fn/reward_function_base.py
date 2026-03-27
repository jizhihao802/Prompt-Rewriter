import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import torch, random, numpy as np


# =============================
# 设备与模型初始化（全局）
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Init] device: {device}")

_model_sbert = SentenceTransformer(
    "/root/autodl-tmp/model/all-mpnet-base-v2",
    device=str(device)
)

_tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/model/qwen3-8b",
    trust_remote_code=True
)

llm = LLM(
    model="/root/autodl-tmp/model/qwen3-8b",
    tensor_parallel_size=1,      
    trust_remote_code=True,
    max_model_len=2048,
    skip_tokenizer_init=False,
    disable_log_stats=True,
    seed=42,
)

# 统一 Sampling 配置
_sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=128,     # 这个会在函数里被动态改写
    n=1,
    logprobs=0,
    ignore_eos=False,
    repetition_penalty=1.0,
)

# =============================
# 工具函数
# =============================
def embed(text: str):
    return _model_sbert.encode(text, normalize_embeddings=True)

def cosine_similarity(a: str, b: str) -> float:
    """返回 a 和 b 的余弦相似度（标量）"""
    try:
        return float(util.cos_sim(embed(a), embed(b)))
    except Exception as e:
        print(f"[Warning] cosine_similarity failed: {e}")
        return 0.0

def generate_local_response(prompt: str, max_new_tokens=0):
    """
    使用 vLLM 生成文本
    vLLM 自动返回“仅生成的部分”，不包含 prompt
    """
    if max_new_tokens == 0:
        max_new_tokens = len(prompt)

    # 修改 vLLM 的 max_tokens
    _sampling_params.max_tokens = max_new_tokens

    # vLLM 只需要 prompt，不需要 tokenizer.encode()
    outputs = llm.generate([prompt], _sampling_params)
    result = outputs[0].outputs[0].text.strip()

    return result


def extract_methods(solution_str: str):
    """
    从模型输出中提取优化方法编号
    支持：
      - '###2/4'
      - '2/4'
      - '选择了方法 2/4'
    返回列表形式，例如 ['2','4']
    """
    match = re.search(r"(?:###\s*)?([\d/]+)", solution_str)
    if match:
        methods = match.group(1).strip()
        return [m for m in methods.split("/") if m.isdigit()]
    return []

def extract_instruction(text: str) -> str:
    """
    从模型输出中提取 <result> ... </result> 内的内容。
    若未找到，返回原文本。
    """
    match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return text.strip()

# =============================
# 核心 Reward 函数（只使用最终回答的sim_score作为reward）
# =============================
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    纯语义 reward：
    - reward = sim_score
    - 不使用 original_sim / baseline_sim
    - 不使用 delta / 权重
    - 不进行任何 shortcut 判断
    """
    #print(f"回答是{solution_str}")

    # ========== 0. 基本校验 ==========
    if extra_info is None or "instruction" not in extra_info:
        print("[Warning] Missing instruction in extra_info.")
        return 0.0

    original_instruction = extra_info["instruction"]

    # ground_truth token 数，用于控制生成长度
    tokens = _tokenizer.encode(ground_truth, add_special_tokens=False)
    max_new_tokens = len(tokens)

    # ========== 1. 解析方法编号 ==========
    methods = extract_methods(solution_str)
    if not methods:
        return 0.0

    # 去重（不惩罚，只防止重复生成）
    methods = list(dict.fromkeys(methods))

    #print(methods)

    # ========== 2. 决定最终使用的 instruction ==========
    method_text_map = extra_info.get("method_mapping", {})
    #print(method_text_map)

    # 是否选择了“不进行修改”（注意：现在 4 只是普通语义之一）
    selected_texts = [method_text_map.get(m, "") for m in methods]
    selected_texts = [t for t in selected_texts if t]

    if not selected_texts or "无需进行优化" in selected_texts:
        # —— 不进行修改：直接使用原始指令 ——
        final_instruction = original_instruction
        #print("无需进行优化")
    else:
        #print("需要进行修改")
        # —— 需要修改：生成新 instruction ——
        modify_methods = "、".join(selected_texts)

        modify_prompt = (
            f"你是一个指令优化器，负责优化用户给到大模型的指令，"
            f"请在保留信息完整的前提下按照{modify_methods}的方法优化"
            f"<instruction>和</instruction>之间的指令。"
            f"<instruction>\n{original_instruction}\n</instruction>"
            f"并按照“<result>你优化后的指令</result>”的格式输出结果，"
            f"并在</result>后终止输出。"
        )

        try:
            modified_instruction = generate_local_response(modify_prompt)
        except Exception as e:
            print(f"[Error] generate modified instruction failed: {e}")
            return 0.0

        if not modified_instruction:
            return 0.0

        final_instruction = extract_instruction(modified_instruction)

    # ========== 3. 用最终 instruction 生成回答 ==========
    try:
        response = generate_local_response(final_instruction, max_new_tokens)
    except Exception as e:
        print(f"[Error] generate response failed: {e}")
        return 0.0

    if not response:
        return 0.0

    # ========== 4. 计算语义相似度（唯一 reward） ==========
    try:
        sim_score = cosine_similarity(response, ground_truth)
        #print(f"相似度为：{sim_score}")
    except Exception as e:
        print(f"[Error] similarity calc failed: {e}")
        sim_score = 0.0

    # safety clamp
    #if sim_score < 0.0:
    #    sim_score = 0.0
    #if sim_score > 1.0:
    #    sim_score = 1.0

    return sim_score
