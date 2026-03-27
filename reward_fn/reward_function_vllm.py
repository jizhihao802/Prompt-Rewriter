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
# 核心 Reward 函数（更新）
# =============================
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    新的 reward 流程（ground_truth 是最终回答）：
      1) 提取方法编号
      2) 生成 modified_instruction
      3) 生成新的回答 response_new
      4) 计算与 ground_truth 的语义相似度 sim_score
      5) 加入改进幅度奖励 sim_delta = sim_score - orig_similarity
         以及相比baseline提升幅度 baseline_delta = sim_score - baseline_sim
         最终 reward = α * sim_score + β * sim_delta + γ * baseline_delta
    """
    #print(f"回答是{solution_str}")
    alpha=0.5
    beta=0.4
    gamma=0.1

    x=0.8
    y=1.2
    z=0.5

    tokens = _tokenizer.encode(ground_truth, add_special_tokens=False)  # 不计入特殊 token
    num_tokens_truth = len(tokens) #统计ground_truth的tokens数目作为回答模型生成回答的max_new_tokens

    # 0. 获取原始 instruction
    if extra_info and "instruction" in extra_info:
        original_instruction = extra_info["instruction"]
    else:
        print("[Warning] Missing original instruction in extra_info.")
        return 0.0

    # 获取原始相似度和baseline相似度
    orig_sim = extra_info.get("orig_similarity", 0.0) if extra_info else 0.0
    baseline_sim = extra_info.get("baseline_similarity", 0.0) if extra_info else 0.0

    # 1. 提取 agent 输出的方法编号
    methods = extract_methods(solution_str)
    
    if not methods:
        #print(f"[Reward] ⚠️ No valid methods found in solution: {solution_str}")
        return 0.0

    if "4" in methods:
        #print("[Reward] Detected method 0 → ❗ 不进行修改，直接计算损失")

        sim_score = orig_sim

        sim_delta = 0.0
        baseline_delta = (sim_score - baseline_sim)/sim_score

        reward = alpha * sim_score + beta * sim_delta + gamma * baseline_delta
        reward = reward * x
        """
        print("========== Reward Debug (NO MODIFICATION MODE) ==========")
        print(f"Methods Chosen: {methods}")
        print(f"Original Instruction Used:\n{original_instruction}")
        print(f"Ground Truth:\n{ground_truth}")
        print(f"Orig Similarity = {orig_sim:.4f}")
        print(f"Baseline Similarity = {baseline_sim:.4f}")
        print(f"New Similarity  = {sim_score:.4f}")
        print(f"Sim Delta       = {sim_delta:+.4f}")
        print(f"Baseline Delta  = {baseline_delta:+.4f}")
        print(f"Final Reward    = {reward:.4f}")
        print("==========================================================")
        """
        other_methods = [m for m in methods if m != "4"]

        if len(other_methods) > 0:
            # 惩罚值（可自定义）
            penalty = -0.05 * len(other_methods)

            #print(f"⚠️ Penalty Added! methods = {methods}, penalty = {penalty}")
            reward += penalty
        
        if reward < 0.0:
            reward = 0.0

        #print(f"Methods Chosen: {methods}")
        return reward

    #print("所选方法不为0，进行指令修改")
    # ================================
    # ★ 去重重复方法 + 惩罚重复项
    # ================================
    original_len = len(methods)
    methods = list(dict.fromkeys(methods))  # 保留顺序去重

    if len(methods) < original_len:
        repeat_count = original_len - len(methods)
        penalty = -0.05 * repeat_count
        #print(f"⚠️ Detected duplicated methods: penalty = {penalty}")
        # 立刻加到reward后面，因此先定义 reward=0（后续会被重新计算）
        reward = penalty
    else:
        reward = 0.0

    # 2. 构造修改说明
    #method_text_map = {
    #    "1": "调整指令结构",
    #    "2": "缩短指令长度",
    #    "3": "增加任务说明"
    #}
    method_text_map = extra_info.get("method_mapping", {})
    #print(method_text_map)

    selected_texts = [method_text_map.get(m, "") for m in methods if m in method_text_map]
    modify_methods = "、".join([t for t in selected_texts if t]) or "不修改"

    # 3. 生成优化后指令i
    modify_prompt = (
        f"你是一个指令优化器，负责优化用户给到大模型的指令，请在保留信息完整的前提下按照{modify_methods}的方法优化<instruction>和</instruction>之间的指令。"
        f"<instruction>\n{original_instruction}\n</instruction>"
        f"并按照“<result>你优化后的指令</result>”的格式输出结果，并在</result>后终止输出。"
    )

    try:
        modified_instruction = generate_local_response(modify_prompt)
    except Exception as e:
        print(f"[Error] generate modified instruction failed: {e}")
        return 0.0

    if not modified_instruction:
        print("[Warning] modified_instruction is empty.")
        return 0.0

    # 4. 用修改后指令生成最终回答
    rewrited_prompt = extract_instruction(modified_instruction)
    try:
        response_new = generate_local_response(rewrited_prompt, num_tokens_truth)
    except Exception as e:
        print(f"[Error] generate final answer failed: {e}")
        return 0.0

    if not response_new:
        print("[Warning] response_new is empty.")
        return 0.0

    # 5. 计算与 ground truth 的相似度
    try:
        sim_score = cosine_similarity(response_new, ground_truth)
    except Exception as e:
        print(f"[Error] similarity calc failed: {e}")
        sim_score = 0.0

    # 6. 计算改进幅度和相比baseline提升幅度
    sim_delta = (sim_score - orig_sim)/sim_score
    baseline_delta = (sim_score - baseline_sim)/sim_score

    # 7. 综合奖励（语义相似度 + 改进幅度）
    reward = alpha * sim_score + beta * sim_delta + gamma * baseline_delta + reward
    
    if sim_score > orig_sim:
        reward = reward * y
    else:
        reward = reward * z
 
    if reward < 0.0:
        reward = 0.0

    #print(f"Methods Chosen: {methods}")

    # 可选：用 tanh 压缩，控制 reward 范围
    #reward = float(torch.tanh(torch.tensor(reward * scale)))
    
    """
    # Debug 输出（训练时可注释）
    print("========== Reward Debug ==========")
    print(f"Methods Chosen: {methods}")
    print(f"Modify Description: {modify_methods}")
    print(f"Modify Prompt:\n{modify_prompt}")
    print(f"Modified Instruction:\n{modified_instruction}")
    print(f"Rewrited Prompt:\n{rewrited_prompt}")
    print(f"New Answer:\n{response_new}")
    print(f"Ground Truth:\n{ground_truth}")
    print(f"Orig Similarity = {orig_sim:.4f}")
    print(f"Baseline Similarity = {baseline_sim:.4f}")
    print(f"New Similarity  = {sim_score:.4f}")
    print(f"Sim Delta       = {sim_delta:+.4f}")
    print(f"Baseline Delta  = {baseline_delta:+.4f}")
    print(f"Final Reward    = {reward:.4f}")
    print("===================================")
    """
    return reward
