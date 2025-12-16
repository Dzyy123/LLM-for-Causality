import json
import os
import numpy as np  

def load_causal_500_json(json_path: str = "./causal_500.json") -> list:
    """Load REAL causal test set (no fake data)"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Test set JSON not found: {json_path}")
    if not json_path.endswith(".json"):
        raise ValueError("❌ Only REAL JSON test set is supported")
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Validate REAL data fields
    required_fields = ["element1", "element2", "causal"]
    for idx, item in enumerate(json_data):
        for field in required_fields:
            if field not in item:
                raise KeyError(f"❌ REAL sample {idx} missing field: {field}")
    
    print(f"✅ Loaded REAL test set: {len(json_data)} samples")
    return json_data

def get_model_probs_for_json(
    json_data: list,
    client,  
    batch_size: int = 20
) -> list:
    """Get REAL LLM probabilities for test set (no fake data)"""
    from whitebox_token_utils import extract_yes_no_probs
    
    expert_probs_list = []
    total_samples = len(json_data)
    
    if total_samples == 0:
        raise ValueError("❌ Empty test set! Cannot generate REAL probabilities.")
    
    for batch_idx in range(0, total_samples, batch_size):
        batch_start = batch_idx
        batch_end = min(batch_idx + batch_size, total_samples)
        batch_data = json_data[batch_start:batch_end]
        
        for item in batch_data:
            e1 = item["element1"]
            e2 = item["element2"]
            prompt = f"Is there any direct causal relationship between {e1} and {e2}? Only output Yes or No."
            system_prompt = """You are a professional causal inference expert. 
Strictly distinguish direct causality from correlation. Only output Yes or No (no other text)."""
            
            # Only REAL LLM response (no fake)
            try:
                response = client.chat(prompt=prompt, system_prompt=system_prompt)
                probs = client.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
                yes_prob, _ = extract_yes_no_probs(probs)
                expert_probs_list.append(yes_prob)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to predict REAL probability for {e1}→{e2}: {e}\nDo NOT use fake data! Fix LLM connection first.")
        
        print(f"📊 Batch {batch_idx//batch_size + 1} completed | Predicted {batch_end}/{total_samples} REAL probabilities")
    
    if len(expert_probs_list) != total_samples:
        raise ValueError(f"❌ REAL probabilities count ({len(expert_probs_list)}) mismatch with samples ({total_samples})")
    
    print(f"✅ Generated REAL probabilities for {len(expert_probs_list)} samples")
    return expert_probs_list