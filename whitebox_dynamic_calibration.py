# whitebox_dynamic_calibration.py
import numpy as np
from whitebox_calibration import temperature_scaling, calculate_entropy

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