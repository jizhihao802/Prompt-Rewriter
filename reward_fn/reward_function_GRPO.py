import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import re
import torch
import torch, random, numpy as np

# =============================
# 工具函数
# =============================
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

# =============================
# 核心 Reward 函数（只使用最终回答的sim_score作为reward）
# =============================
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    """
    离线 oracle reward（可调试版）
    """
    debug = True #控制是否输出debug内容

    def dprint(*args):
        if debug:
            print(*args)

    # ========= 0. 基本校验 =========
    if extra_info is None:
        dprint("[Reward] extra_info is None")
        return 0.0

    oracle_scores = extra_info.get("oracle_method_scores")
    if not isinstance(oracle_scores, dict):
        dprint("[Reward] oracle_method_scores missing or invalid")
        return 0.0

    # ========= 1. 解析方法 =========
    methods = extract_methods(solution_str)

    if not methods:
        dprint("[Reward] No methods parsed from solution_str")
        dprint("  solution_str =", solution_str)
        return 0.0

    methods_set = set(methods)

    dprint(f"[Reward] Parsed methods = {methods}")

    # ========= 2. 查找 oracle =========
    for combo_id, info in oracle_scores.items():
        combo_methods = info.get("methods", [])
        if set(combo_methods) == methods_set:
            sim = float(info.get("sim_score", 0.0))

            dprint(
                f"[Reward] HIT oracle combo={combo_id} "
                f"methods={combo_methods} sim={sim:.4f}"
            )
            return sim

    # ========= 3. 未命中 =========
    dprint("[Reward] MISS oracle")
    dprint("  parsed methods =", sorted(methods_set))
    dprint("  oracle combos  =",
           [v["methods"] for v in oracle_scores.values()])

    return 0.0
