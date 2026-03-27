import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

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

_model_local = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/model/qwen3-8b",
    device_map={"": device},
    dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

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
    使用本地 LM 生成文本，并只返回模型新生成的部分（去掉输入 prompt）
    """
    # 编码输入
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    if max_new_tokens == 0:
        max_new_tokens=len(prompt)

    with torch.no_grad():
        # 生成序列
        outputs = _model_local.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )

    # 解码整个输出（包含输入 + 新生成部分）
    full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ====== 去掉 prompt 对应部分，只保留新生成内容 ======
    prompt_text = _tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
    if full_text.startswith(prompt_text):
        new_text = full_text[len(prompt_text):].strip()
    else:
        new_text = full_text  # 如果没有匹配，直接返回全部

    return new_text


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
    alpha=0.5
    beta=0.3
    gamma=0.2

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
        print(f"[Reward] ⚠️ No valid methods found in solution: {solution_str}")
        return 0.0

    if "4" in methods:
        #print("[Reward] Detected method 0 → ❗ 不进行修改，直接计算损失")

        sim_score = orig_sim

        sim_delta = sim_score - orig_sim
        baseline_delta = sim_score - baseline_sim

        reward = alpha * sim_score + beta * sim_delta + gamma * baseline_delta
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
        return reward

    #print("所选方法不为0，进行指令修改")
    # 2. 构造修改说明
    method_text_map = {
        "1": "调整指令结构",
        "2": "缩短指令长度",
        "3": "增加任务说明"
    }
    selected_texts = [method_text_map.get(m, "") for m in methods if m in method_text_map]
    modify_methods = "、".join([t for t in selected_texts if t]) or "不修改"

    # 3. 生成优化后指令i
    modify_prompt = (
        f"你是一个指令优化器，请在保留信息完整的前提下按照{modify_methods}的方法优化<instruction>和</instruction>之间的指令。"
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
    sim_delta = sim_score - orig_sim
    baseline_delta = sim_score - baseline_sim

    # 7. 综合奖励（语义相似度 + 改进幅度）
    reward = alpha * sim_score + beta * sim_delta + gamma * baseline_delta

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
