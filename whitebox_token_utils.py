# whitebox_token_utils.py
import numpy as np

def extract_yes_no_probs(token_dist: list) -> tuple:
    """从token分布中提取Yes/No的聚合概率（适配yes/no输出）"""
    if not token_dist:
        return 0.5, 0.5  # 默认值
    
    yes_prob = 0.0
    no_prob = 0.0
    # 遍历所有token，聚合Yes/No相关概率
    for token_dict in token_dist:
        for token, prob in token_dict.items():
            token_lower = token.lower()
            if token_lower in ["yes", "y"]:
                yes_prob += prob
            elif token_lower in ["no", "n"]:
                no_prob += prob
    
    # 归一化
    total = yes_prob + no_prob
    if total > 0:
        yes_prob /= total
        no_prob /= total
    else:
        yes_prob, no_prob = 0.5, 0.5
    
    return yes_prob, no_prob

def extract_abcd_probs(token_dist: list) -> tuple:
    """从token分布中提取A/B/C/D的概率（适配多选项输出）"""
    if not token_dist:
        return 0.25, 0.25, 0.25, 0.25  # 默认值
    
    a_prob, b_prob, c_prob, d_prob = 0.0, 0.0, 0.0, 0.0
    for token_dict in token_dist:
        for token, prob in token_dict.items():
            token_upper = token.upper()
            if token_upper == "A":
                a_prob += prob
            elif token_upper == "B":
                b_prob += prob
            elif token_upper == "C":
                c_prob += prob
            elif token_upper == "D":
                d_prob += prob
    
    # 归一化
    total = a_prob + b_prob + c_prob + d_prob
    if total > 0:
        a_prob /= total
        b_prob /= total
        c_prob /= total
        d_prob /= total
    else:
        a_prob, b_prob, c_prob, d_prob = 0.25, 0.25, 0.25, 0.25
    
    return a_prob, b_prob, c_prob, d_prob