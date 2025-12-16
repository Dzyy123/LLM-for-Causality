# whitebox_calibration.py（补充NLL）
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

def temperature_scaling(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    温度缩放校准：调整概率分布的锐度
    - probs: 原始概率数组（如[0.8, 0.2]）
    - temperature: 温度系数（<1锐化，>1平滑）
    返回校准后的概率（归一化）
    """
    # 避免log(0)，加极小值
    probs = np.clip(probs, 1e-10, 1.0)
    # 转换为logits
    logits = np.log(probs)
    # 温度缩放
    scaled_logits = logits / temperature
    # 转回概率
    exp_logits = np.exp(scaled_logits)
    calibrated_probs = exp_logits / np.sum(exp_logits)
    return calibrated_probs

def calculate_mse(true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
    """计算真实概率与预测概率的MSE（均方误差）"""
    return mean_squared_error(true_probs.flatten(), pred_probs.flatten())

def calculate_nll(true_labels: np.ndarray, pred_probs: np.ndarray) -> float:
    """计算负对数似然（Negative Log-Likelihood）：越小表示预测越准（对应你的“负对数函数”）"""
    # 避免log(0)，加极小值
    pred_probs = np.clip(pred_probs, 1e-10, 1.0)
    # 二分类：取真实标签对应的概率的对数
    log_probs = np.log(pred_probs[np.arange(len(true_labels)), true_labels.astype(int)])
    nll = -np.mean(log_probs)
    return nll

def calculate_entropy(probs: np.ndarray) -> float:
    """计算概率分布的熵值（衡量不确定性，熵越小越确定）"""
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log2(probs)) / len(probs)
    return entropy

def calibrate_expert_prob(yes_prob: float, no_prob: float, opt_temp: float = 1.0) -> tuple:
    """
    校准单个专家的Yes/No概率
    返回：(校准后的yes_prob, 校准后的no_prob, 熵值)
    """
    # 构造概率数组
    raw_probs = np.array([yes_prob, no_prob])
    # 温度缩放校准
    calibrated_probs = temperature_scaling(raw_probs, opt_temp)
    # 计算熵值
    entropy = calculate_entropy(calibrated_probs)
    return calibrated_probs[0], calibrated_probs[1], entropy

def find_optimal_temperature(
    expert_probs_list: list, 
    true_labels: list = None,
    optimize_target: str = "nll"  # 可选："mse"（均方误差）/ "nll"（负对数似然）
) -> float:
    """
    遍历温度范围，找到MSE/NLL最小的最优温度（基于causal_500.json的真实标签）
    - expert_probs_list: 所有样本的原始Yes概率列表（来自causal_500.json的每个样本）
    - true_labels: 所有样本的真实标签（1=Yes，0=No）（来自causal_500.json的causal字段）
    - optimize_target: 优化目标（mse/nll）
    """
    if true_labels is None or len(expert_probs_list) != len(true_labels):
        return 1.0  # 无真实标签时用默认温度
    
    # 遍历温度范围（可调整）
    temp_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    loss_list = []
    
    for temp in temp_range:
        # 校准所有样本的概率
        calibrated_probs = []
        for yes_prob in expert_probs_list:
            raw_probs = np.array([yes_prob, 1-yes_prob])
            calib_probs = temperature_scaling(raw_probs, temp)
            calibrated_probs.append(calib_probs[0])  # 取Yes的概率
        
        # 计算当前温度下的损失（MSE或NLL）
        if optimize_target == "mse":
            loss = calculate_mse(np.array(true_labels), np.array(calibrated_probs))
        elif optimize_target == "nll":
            # NLL需要构造二分类概率矩阵 + 真实标签索引
            true_labels_np = np.array(true_labels)
            calib_probs_np = np.array([[p, 1-p] for p in calibrated_probs])
            loss = calculate_nll(true_labels_np, calib_probs_np)
        else:
            loss = calculate_mse(np.array(true_labels), np.array(calibrated_probs))
        
        loss_list.append(loss)
    
    # 找损失最小的温度（最优温度）
    opt_temp = temp_range[np.argmin(loss_list)]
    # 可选：打印各温度的损失（便于调试）
    print(f"【最优温度计算】温度范围: {temp_range}")
    print(f"【最优温度计算】{optimize_target}列表: {[round(l, 4) for l in loss_list]}")
    print(f"【最优温度计算】最优温度: {opt_temp} (最小{optimize_target}: {min(loss_list):.4f})")
    return opt_temp