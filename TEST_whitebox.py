import numpy as np
import random
from whitebox_calibration import (
    temperature_scaling,
    calculate_mse,
    calculate_entropy,
    calibrate_expert_prob,
    find_optimal_temperature,
    calculate_nll
)
# 导入Token提取函数
from whitebox_token_utils import extract_yes_no_probs, extract_abcd_probs
from whitebox_json_loader import load_causal_500_json, get_model_probs_for_json
from llm_utils import LocalLLMClient, ChatMessage, setup_logging

# ============== Configuration ==============
# MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"  # HuggingFace model identifier
# LOCAL_PATH = "./models/qwen3-4b-thinking"  # Local storage path
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # HuggingFace model identifier
LOCAL_PATH = "./models/qwen3-4b-instruct"  # Local storage path
MIRROR_URL = "https://hf-mirror.com"  # Use mirror for faster download in China (set to None for official HF)
DEVICE = "cpu"  # "auto", "cuda", or "cpu"
JSON_PATH = "./causal_500.json"
OPTIMIZE_TARGET = "nll"

# ============== 第一步：读取JSON + 批量获取模型预测概率 ==============
print("="*80)
print("Step 1: 读取causal_500.json并批量获取模型预测概率")
print("="*80)
# 读取JSON
json_data = load_causal_500_json(JSON_PATH)

# ========== 关键修改：随机抽取50个样本（替代取前50个） ==========
# 容错处理：如果数据总量不足50，取全部并提示
sample_size = 50
if len(json_data) < sample_size:
    print(f"⚠️  注意：JSON数据总量({len(json_data)})不足{sample_size}个，将使用全部数据")
    random_json_data = json_data  # 不足时取全部
else:
    # 随机抽取50个（无重复）
    random_json_data = random.sample(json_data, sample_size)
# 赋值给原变量，保持后续逻辑不变
json_data = random_json_data
# ==============================================================

# 批量获取每个样本的Yes概率（耗时，因为要逐个调用模型）
expert_probs_list = get_model_probs_for_json(
    json_data=json_data,
    model_id=MODEL_ID,
    local_path=LOCAL_PATH,
    device=DEVICE
)
# 提取JSON中的真实标签（causal字段）
true_labels = [item["causal"] for item in json_data]

# ============== 第二步：基于JSON数据集计算最优温度 ==============
print("\n" + "="*80)
print("Step 2: 基于causal_500.json计算最优温度（优化目标：{}）".format(OPTIMIZE_TARGET))
print("="*80)
opt_temp = find_optimal_temperature(
    expert_probs_list=expert_probs_list,
    true_labels=true_labels,
    optimize_target=OPTIMIZE_TARGET
)
print(f"✅ 基于causal_500.json的最优温度: {opt_temp}")

# ============== 第三步：用最优温度测试单个案例（验证效果） ==============
print("\n" + "="*80)
print("Step 3: 用最优温度测试单个案例（吸烟→肺癌）")
print("="*80)
# 重新初始化模型客户端（用于单案例测试）
setup_logging(level="INFO")
CLIENT = LocalLLMClient(
    model_id=MODEL_ID,
    local_path=LOCAL_PATH,
    mirror_url="https://hf-mirror.com",
    device=DEVICE,
    max_tokens=4000,
)


# TEST 1: Smoking -> Lung Cancer

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and lung cancer?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
print(f"Response: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
yes_prob, no_prob = extract_yes_no_probs(probs)
print(f"原始概率 → Yes: {yes_prob:.4f}, No: {no_prob:.4f}")

# 用最优温度校准
calib_yes, calib_no, calib_entropy = calibrate_expert_prob(yes_prob, no_prob, opt_temp)
# 计算MSE/NLL（真实标签：吸烟→肺癌的causal=1）
true_label = 1
raw_probs = np.array([yes_prob, no_prob])
true_probs = np.array([true_label, 1-true_label])
mse = calculate_mse(true_probs, raw_probs)
calib_probs = np.array([calib_yes, calib_no])
calib_mse = calculate_mse(true_probs, calib_probs)
# 计算NLL
nll = calculate_nll(np.array([true_label]), np.array([raw_probs]))
calib_nll = calculate_nll(np.array([true_label]), np.array([calib_probs]))

# 输出校准对比
print(f"\n=== 最优温度校准结果 ===")
print(f"最优温度: {opt_temp}")
print(f"校准后概率 → Yes: {calib_yes:.4f}, No: {calib_no:.4f}")
print(f"原始熵值: {calculate_entropy(raw_probs):.4f} → 校准后熵值: {calib_entropy:.4f}")
print(f"原始MSE: {mse:.4f} → 校准后MSE: {calib_mse:.4f}")
print(f"原始NLL: {nll:.4f} → 校准后NLL: {calib_nll:.4f}")

# TEST 2: Smoking -> Malaria

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and malaria?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
print(f"Response: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

# TEST 3: Smoking -> Pulmonary Edema

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and pulmonary edema?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
print(f"Response: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

# TEST 4: Smoking -> Pulmonary Edema with Check, multiple selection choice

conv = CLIENT.start_conversation(
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, according to your confidence in the answer, answer **in ONLY one letter**: \"A\" for absolutely yes, \"B\" for probably yes, \"C\" for probably no, and \"D\" for absolutely no.",
    max_history=10  # Keep last 10 messages
)

response = conv.send("Is there any causal relationship between smoking and pulmonary edema?")
print(f"Turn 1: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

response = conv.send("I think the answer is yes. Answer me, are you firmly sure about your previous answer?")
print(f"Turn 2: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

CLIENT.unload_model()