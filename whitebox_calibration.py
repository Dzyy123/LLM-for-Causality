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

def dynamic_calibrate(
    yes_prob: float,
    no_prob: float,
    performance_stats: dict,
    base_temp: float = 1.0
) -> tuple:
    """
    基于相似样本表现动态校准置信度
    核心逻辑：
    - 相似样本极端错误率越高 → 温度越高（置信度下调越多）
    - 相似样本极端正确率越高 → 温度越低（置信度微调/不调）
    - 无相似样本 → 用基础温度
    """
    extreme_error_ratio = performance_stats["extreme_error_ratio"]
    extreme_correct_ratio = performance_stats["extreme_correct_ratio"]
    avg_similarity = performance_stats["avg_similarity"]
    
    # 动态计算温度系数（相似度越高，调整幅度越大）
    temp_adjust_coeff = extreme_error_ratio * 2.0  # 极端错误率最高让温度+2
    temp_adjust_coeff -= extreme_correct_ratio * 0.5  # 极端正确率最高让温度-0.5
    temp_adjust_coeff *= avg_similarity  # 相似度加权（低相似则调整幅度小）
    
    # 最终温度（限制范围：0.8~3.0，避免极端）
    final_temp = base_temp + temp_adjust_coeff
    final_temp = np.clip(final_temp, 0.8, 3.0)
    
    # 温度缩放校准
    raw_probs = np.array([yes_prob, no_prob])
    calibrated_probs = temperature_scaling(raw_probs, final_temp)
    calib_entropy = calculate_entropy(calibrated_probs)
    
    # 输出校准逻辑解释
    print(f"\n🔧 动态校准逻辑：")
    print(f"  相似样本极端错误率：{extreme_error_ratio:.2%} → 温度+{temp_adjust_coeff:.2f}")
    print(f"  相似样本极端正确率：{extreme_correct_ratio:.2%} → 温度-{extreme_correct_ratio*0.5:.2f}")
    print(f"  平均相似度：{avg_similarity:.4f} → 调整幅度加权")
    print(f"  最终校准温度：{final_temp:.2f}（基础温度：{base_temp}）")
    
    return calibrated_probs[0], calibrated_probs[1], calib_entropy, final_temp


def identify_extreme_error_samples(expert_probs_list: list, true_labels: list) -> dict:
    """Identify extreme error/correct samples in test set (辅助函数)"""
    extreme_error_samples = []  # Extreme error: prob≥0.9 but label=0, or prob≤0.1 but label=1
    extreme_correct_samples = []# Extreme correct: prob≥0.9 and label=1, or prob≤0.1 and label=0
    normal_samples = []         # Normal samples: prob 0.1~0.9
    
    for idx, (prob, label) in enumerate(zip(expert_probs_list, true_labels)):
        if (prob >= 0.9 and label == 0) or (prob <= 0.1 and label == 1):
            extreme_error_samples.append({"index": idx, "prob": prob, "label": label})
        elif (prob >= 0.9 and label == 1) or (prob <= 0.1 and label == 0):
            extreme_correct_samples.append({"index": idx, "prob": prob, "label": label})
        else:
            normal_samples.append({"index": idx, "prob": prob, "label": label})
    
    return {
        "extreme_error_samples": extreme_error_samples,
        "extreme_correct_samples": extreme_correct_samples,
        "normal_samples": normal_samples,
        "stats": {
            "extreme_error_ratio": len(extreme_error_samples)/len(expert_probs_list),
            "extreme_correct_ratio": len(extreme_correct_samples)/len(expert_probs_list),
            "normal_ratio": len(normal_samples)/len(expert_probs_list)
        }
    }

def calculate_entropy(probs: np.ndarray) -> float:
    """Calculate entropy (uncertainty) of probability distribution (辅助函数)"""
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

def temperature_scaling_base(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """基础温度缩放（辅助函数）"""
    probs = probs / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))
    return probs

# ===================== 方法1：加权误差修正法 =====================
def weighted_error_calibration(
    new_init_prob: float,  
    similar_samples: list, 
    test_embedding_db: dict 
) -> float:
    """
    加权误差修正法：用相似样本的误差（真实值-预测值）加权修正初始概率
    """
    if not similar_samples:
        print("  ⚠️ No similar samples, return original probability")
        return new_init_prob
    
    # 提取相似样本数据
    sim_sample_ids = [s["sample_id"] for s in similar_samples]
    sim_scores = [s["similarity"] for s in similar_samples]
    sim_errors = test_embedding_db["sample_errors"][sim_sample_ids]
    
    # 加权平均误差
    total_sim = sum(sim_scores)
    if total_sim == 0:
        return new_init_prob
    weighted_error = sum([s * e for s, e in zip(sim_scores, sim_errors)]) / total_sim
    
    # 校准并裁剪
    calibrated_prob = np.clip(new_init_prob + weighted_error, 0.0, 1.0)
    
    # 输出过程
    print(f"\n🔍 [Weighted Error Correction]")
    print(f"  Original Prob: {new_init_prob:.4f} | Weighted Error: {weighted_error:.4f} | Calibrated Prob: {calibrated_prob:.4f}")
    return calibrated_prob

# ===================== 方法2：相似样本置信度融合法 =====================
def similarity_confidence_fusion(
    new_init_prob: float,
    similar_samples: list,
    test_embedding_db: dict
) -> float:
    """
    优化版：过滤极端错误样本 + 动态融合系数 + 误差修正兜底
    """
    if not similar_samples:
        print("  ⚠️ No similar samples, return original probability")
        return new_init_prob
    
    # ========== 步骤1：过滤极端错误样本（只保留正确/正常样本） ==========
    # 筛选条件：排除 performance=0（Extreme Error）的样本
    valid_similar_samples = [s for s in similar_samples if s["performance"] != 0]
    if not valid_similar_samples:
        print("  ⚠️ All similar samples are Extreme Error! Use error correction instead.")
        # 兜底：改用加权误差修正（用误差而非预测概率）
        sim_sample_ids = [s["sample_id"] for s in similar_samples]
        sim_scores = [s["similarity"] for s in similar_samples]
        sim_errors = test_embedding_db["sample_errors"][sim_sample_ids]
        total_sim = sum(sim_scores)
        weighted_error = sum([s * e for s, e in zip(sim_scores, sim_errors)]) / total_sim if total_sim > 0 else 0
        calibrated_prob = np.clip(new_init_prob + weighted_error, 0.0, 1.0)
        print(f"  Error-corrected Prob: {calibrated_prob:.4f}")
        return calibrated_prob
    
    # ========== 步骤2：提取有效样本的相似度和预测概率 ==========
    sim_sample_ids = [s["sample_id"] for s in valid_similar_samples]
    sim_scores = [s["similarity"] for s in valid_similar_samples]
    sim_probs = test_embedding_db["sample_pred_probs"][sim_sample_ids]
    
    # ========== 步骤3：动态计算融合系数（根据有效样本的正确率） ==========
    # 计算有效样本的正确率（performance=1的样本占比）
    correct_count = len([s for s in valid_similar_samples if s["performance"] == 1])
    valid_total = len(valid_similar_samples)
    correct_ratio = correct_count / valid_total if valid_total > 0 else 0.0
    
    # 动态融合系数：正确率越高，融合系数越大（范围：0.1~0.8）
    fusion_coeff = max(0.1, min(0.8, correct_ratio))  # 避免系数过0或过1
    original_coeff = 1 - fusion_coeff
    
    # ========== 步骤4：有效样本的相似度加权融合 ==========
    total_sim = sum(sim_scores)
    fused_prob = sum([s * p for s, p in zip(sim_scores, sim_probs)]) / total_sim if total_sim > 0 else new_init_prob
    
    # ========== 步骤5：动态加权融合（原始概率 + 有效样本融合概率） ==========
    calibrated_prob = np.clip(
        original_coeff * new_init_prob + fusion_coeff * fused_prob,
        0.0, 1.0
    )
    
    # 输出优化后的过程
    print(f"\n🔍 [Optimized Similarity Confidence Fusion]")
    print(f"  Valid Similar Samples: {valid_total}/{len(similar_samples)} (filtered Extreme Error)")
    print(f"  Similar Samples Correct Ratio: {correct_ratio:.2%} → Fusion Coeff: {fusion_coeff:.2f}")
    print(f"  Original Prob: {new_init_prob:.4f} | Fused Prob (Valid Samples): {fused_prob:.4f}")
    print(f"  Calibrated Prob: {calibrated_prob:.4f} (original_coeff={original_coeff:.2f}, fusion_coeff={fusion_coeff:.2f})")
    
    return calibrated_prob

# ===================== 方法3：不确定性加权校准法 =====================
def uncertainty_weighted_calibration(
    new_init_prob: float,
    similar_samples: list,
    test_embedding_db: dict
) -> float:
    """
    不确定性加权校准法：结合相似样本的熵（不确定性），高不确定性样本赋予更高的误差修正权重
    """
    if not similar_samples:
        print("  ⚠️ No similar samples, return original probability")
        return new_init_prob
    
    # 提取相似样本数据
    sim_sample_ids = [s["sample_id"] for s in similar_samples]
    sim_scores = [s["similarity"] for s in similar_samples]
    sim_errors = test_embedding_db["sample_errors"][sim_sample_ids]
    
    # 计算相似样本的熵（不确定性）
    sim_probs = test_embedding_db["sample_pred_probs"][sim_sample_ids]
    sim_entropies = [calculate_entropy(np.array([p, 1-p])) for p in sim_probs]
    
    # 不确定性加权（熵越高，权重越大）
    entropy_weights = [e / sum(sim_entropies) if sum(sim_entropies) > 0 else 1/len(sim_entropies) for e in sim_entropies]
    weighted_sim_scores = [s * w for s, w in zip(sim_scores, entropy_weights)]
    
    # 加权误差修正
    total_weighted_sim = sum(weighted_sim_scores)
    if total_weighted_sim == 0:
        return new_init_prob
    weighted_error = sum([s * e for s, e in zip(weighted_sim_scores, sim_errors)]) / total_weighted_sim
    
    # 校准并裁剪
    calibrated_prob = np.clip(new_init_prob + weighted_error, 0.0, 1.0)
    
    # 输出过程
    print(f"\n🔍 [Uncertainty Weighted Calibration]")
    print(f"  Original Prob: {new_init_prob:.4f} | Uncertainty-Weighted Error: {weighted_error:.4f} | Calibrated Prob: {calibrated_prob:.4f}")
    return calibrated_prob

# ===================== 方法4：分位数校准法 =====================
def quantile_calibration(
    new_init_prob: float,
    similar_samples: list,
    test_embedding_db: dict,
    quantile: float = 0.95  # 分位数（如0.95表示取95%分位数）
) -> float:
    """
    分位数校准法：基于测试集误差的分位数，限制极端误差影响，调整新样本概率
    """
    if not similar_samples:
        print("  ⚠️ No similar samples, return original probability")
        return new_init_prob
    
    # 提取相似样本误差
    sim_sample_ids = [s["sample_id"] for s in similar_samples]
    sim_errors = test_embedding_db["sample_errors"][sim_sample_ids]
    
    # 计算误差的分位数，限制极端误差
    error_upper = np.quantile(sim_errors, quantile)
    error_lower = np.quantile(sim_errors, 1 - quantile)
    clipped_errors = [np.clip(e, error_lower, error_upper) for e in sim_errors]
    
    # 平均修正误差
    avg_clipped_error = np.mean(clipped_errors)
    calibrated_prob = np.clip(new_init_prob + avg_clipped_error, 0.0, 1.0)
    
    # 输出过程
    print(f"\n🔍 [Quantile Calibration (q={quantile})]")
    print(f"  Original Prob: {new_init_prob:.4f} | Clipped Avg Error: {avg_clipped_error:.4f} | Calibrated Prob: {calibrated_prob:.4f}")
    return calibrated_prob

# ===================== 方法5：个性化温度缩放 =====================
def personalized_temperature_scaling(
    new_init_prob: float,
    similar_samples: list,
    test_embedding_db: dict
) -> float:
    """
    个性化温度缩放：基于相似样本的概率分布，动态计算温度系数，缩放新样本概率
    """
    if not similar_samples:
        print("  ⚠️ No similar samples, return original probability (temp=1.0)")
        return new_init_prob
    
    # 提取相似样本的预测概率
    sim_sample_ids = [s["sample_id"] for s in similar_samples]
    sim_probs = test_embedding_db["sample_pred_probs"][sim_sample_ids]
    sim_true_labels = test_embedding_db["sample_true_labels"][sim_sample_ids]
    
    # 计算相似样本的温度系数（最小化交叉熵）
    def cross_entropy_loss(temp):
        scaled_probs = temperature_scaling_base(np.array([[p, 1-p] for p in sim_probs]), temp)
        ce = -np.sum([sim_true_labels[i] * np.log(scaled_probs[i][0]) + (1 - sim_true_labels[i]) * np.log(scaled_probs[i][1]) for i in range(len(sim_probs))])
        return ce
    
    # 优化温度系数（范围：0.1~10.0）
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(cross_entropy_loss, bounds=(0.1, 10.0), method='bounded')
    optimal_temp = res.x if res.success else 1.0
    
    # 应用温度缩放
    scaled_probs = temperature_scaling_base(np.array([new_init_prob, 1 - new_init_prob]), optimal_temp)
    calibrated_prob = scaled_probs[0]
    
    # 输出过程
    print(f"\n🔍 [Personalized Temperature Scaling]")
    print(f"  Original Prob: {new_init_prob:.4f} | Optimal Temp: {optimal_temp:.2f} | Calibrated Prob: {calibrated_prob:.4f}")
    return calibrated_prob

# ===================== 校准方法选择器 =====================
def calibrate_probability(
    method: str,
    new_init_prob: float,
    similar_samples: list,
    test_embedding_db: dict,
    **kwargs
) -> float:
    """
    校准方法统一入口：通过method参数选择使用哪种校准方法
    :param method: 可选值：weighted_error / similarity_fusion / uncertainty_weighted / quantile / personalized_temp
    :param new_init_prob: 新样本初始概率
    :param similar_samples: 相似样本列表
    :param test_embedding_db: 测试集向量库
    :param kwargs: 其他参数（如quantile校准的quantile值）
    :return: 校准后的概率
    """
    method_map = {
        "weighted_error": weighted_error_calibration,
        "similarity_fusion": similarity_confidence_fusion,
        "uncertainty_weighted": uncertainty_weighted_calibration,
        "quantile": quantile_calibration,
        "personalized_temp": personalized_temperature_scaling
    }
    
    if method not in method_map:
        raise ValueError(f"❌ Invalid calibration method: {method}\nSupported methods: {list(method_map.keys())}")
    
    # 调用对应校准方法
    if method == "quantile":
        quantile = kwargs.get("quantile", 0.95)
        return method_map[method](new_init_prob, similar_samples, test_embedding_db, quantile=quantile)
    else:
        return method_map[method](new_init_prob, similar_samples, test_embedding_db)