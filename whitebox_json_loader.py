# whitebox_json_loader.py
import json
from llm_utils import LocalLLMClient, setup_logging

def load_causal_500_json(json_path: str) -> list:
    """读取causal_500.json，返回样本列表：[{"element1": "...", "element2": "...", "causal": 0/1}, ...]"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 校验字段
    for idx, item in enumerate(data):
        if not all(k in item for k in ["element1", "element2", "causal"]):
            raise ValueError(f"JSON样本{idx}缺少字段：element1/element2/causal")
        # 确保causal是0/1
        item["causal"] = int(item["causal"])
    return data

def get_model_probs_for_json(
    json_data: list,
    model_id: str,
    local_path: str,
    device: str = "cpu"
) -> list:
    """
    对causal_500.json的每个样本，获取模型预测的Yes概率
    返回：所有样本的Yes概率列表（用于后续最优温度计算）
    """
    # 初始化模型客户端
    setup_logging(level="INFO")
    client = LocalLLMClient(
        model_id=model_id,
        local_path=local_path,
        mirror_url="https://hf-mirror.com",
        device=device,
        max_tokens=4000,
    )
    
    expert_probs_list = []  # 存储所有样本的Yes概率
    system_prompt = "You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
    
    # 遍历JSON中的每个样本
    for idx, item in enumerate(json_data):
        element1 = item["element1"]
        element2 = item["element2"]
        prompt = f"Is there any causal relationship between {element1} and {element2}?"
        
        print(f"\n【批量预测】处理样本{idx+1}/{len(json_data)}: {element1} → {element2}")
        # 获取模型响应
        response = client.chat(prompt=prompt, system_prompt=system_prompt)
        # 提取Yes/No概率（复用之前的Token提取函数）
        from whitebox_token_utils import extract_yes_no_probs
        probs = client.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
        yes_prob, no_prob = extract_yes_no_probs(probs)
        expert_probs_list.append(yes_prob)
        print(f"【批量预测】样本{idx+1}的Yes概率: {yes_prob:.4f}")
    
    # 卸载模型
    client.unload_model()
    return expert_probs_list