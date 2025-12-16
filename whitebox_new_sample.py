import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Validate imports (REAL modules only)
try:
    import whitebox_similarity
    print("✅ Imported whitebox_similarity (REAL module)")
    if hasattr(whitebox_similarity, 'find_similar_test_samples'):
        from whitebox_similarity import find_similar_test_samples
    else:
        raise ImportError("❌ find_similar_test_samples not found!")
except ImportError as e:
    raise ImportError(f"❌ Import failed: {e}")

from whitebox_embedding import init_embedding_model, generate_embedding, load_test_embedding_db
from whitebox_calibration import calibrate_probability, calculate_entropy
from whitebox_token_utils import extract_yes_no_probs
from llm_utils import LocalLLMClient  # 保持你原有LLM客户端

# ========== REAL configuration (no fake) ==========
TEST_EMBEDDING_DB_PATH = "test_embedding_db.npz"
TOP_K_SIMILAR = 5
EMBEDDING_LANGUAGE = "en"
LOCAL_EMBEDDING_PATH = "./models/all-MiniLM-L6-v2"
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_LOCAL_PATH = "./models/qwen3-4b-instruct"
DEVICE = "cpu"

# ========== 校准方法配置（可修改此处选择方法） ==========
CALIBRATION_METHOD = "weighted_error"  # 可选值：weighted_error / similarity_fusion / uncertainty_weighted / quantile / personalized_temp
QUANTILE_VALUE = 0.95  # 分位数校准法的参数（仅当method=quantile时生效）

# ========== Initialize REAL models (only once) ==========
# 1. Load REAL Embedding model
embedding_model = init_embedding_model(
    language=EMBEDDING_LANGUAGE,
    local_model_path=LOCAL_EMBEDDING_PATH
)

# 2. Load saved REAL embedding DB (skip regeneration)
test_embedding_db = load_test_embedding_db(TEST_EMBEDDING_DB_PATH)

# 3. Initialize REAL LLM (for new sample prediction)
print("✅ Initialize REAL LLM client (no fake data)")
try:
    llm_client = LocalLLMClient(
        model_id=QWEN_MODEL_ID,
        local_path=QWEN_LOCAL_PATH,
        device=DEVICE,
        max_tokens=4000
    )
except Exception as e:
    raise RuntimeError(f"❌ LLM initialization failed: {e}\nFix LLM first (no fake data)!")

# ========== REAL prediction with selectable calibration ==========
def predict_new_sample_semantic(e1: str, e2: str) -> dict:
    """
    Predict new sample with REAL LLM + selectable calibration method (no fake)
    """
    print("="*80)
    print(f"🆕 New Sample (REAL prediction | Calibration: {CALIBRATION_METHOD})")
    print(f"  Causal Pair: {e1} → {e2}")
    print("="*80)

    # Step 1: Generate REAL semantic vector
    print("Step 1: Generate REAL semantic vector (Embedding model)")
    new_vec = generate_embedding(e1, e2, embedding_model)

    # Step 2: Find REAL similar test samples
    print("Step 2: Match REAL similar test samples")
    similarity_result = find_similar_test_samples(
        new_vec=new_vec,
        test_embedding_db=test_embedding_db,
        top_k=TOP_K_SIMILAR,
        sim_threshold=0.5
    )
    similar_samples = similarity_result["similar_samples"]
    performance_stats = similarity_result["performance_stats"]

    # Print REAL similar samples
    print(f"\n📊 REAL Similar Test Samples (Top-{TOP_K_SIMILAR}):")
    if similar_samples:
        for idx, s in enumerate(similar_samples):
            print(f"  {idx+1}. {s['sample_text']} | Similarity: {s['similarity']} | 模型预测Yes概率: {s['model_pred_yes_prob']} | 真实答案: {s['true_answer']}")
    else:
        print("  No high-similarity REAL test samples (similarity < 0.5)")
    
    print(f"\n📈 REAL Similar Samples Statistics:")

    # Step 3: Get REAL LLM prediction (no fake)
    print("\nStep 3: Get REAL LLM causal prediction (no fake data)")
    prompt = f"Is there any direct causal relationship between {e1} and {e2}? Only output Yes or No."
    system_prompt = """You are a professional causal inference expert. 
Strictly distinguish direct causality from correlation. Only output Yes or No (no other text)."""
    
    # REAL LLM response (no fake)
    try:
        response = llm_client.chat(prompt=prompt, system_prompt=system_prompt)
        raw_response = response.crop_thinking().content.strip()
        probs = llm_client.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
        yes_prob, no_prob = extract_yes_no_probs(probs)
    except Exception as e:
        raise RuntimeError(f"❌ REAL LLM prediction failed: {e}\nFix LLM first (no fake data)!")
    
    raw_entropy = calculate_entropy(np.array([yes_prob, no_prob]))
    print(f"  REAL LLM Output: {raw_response}")
    print(f"  REAL LLM Probability: Yes={yes_prob:.4f}, No={no_prob:.4f} (Entropy: {raw_entropy:.4f})")

    # Step 4: Calibrate probability (selectable method)
    print("\nStep 4: Probability Calibration")
    if CALIBRATION_METHOD == "quantile":
        calib_yes = calibrate_probability(
            method=CALIBRATION_METHOD,
            new_init_prob=yes_prob,
            similar_samples=similar_samples,
            test_embedding_db=test_embedding_db,
            quantile=QUANTILE_VALUE  # 分位数参数
        )
    else:
        calib_yes = calibrate_probability(
            method=CALIBRATION_METHOD,
            new_init_prob=yes_prob,
            similar_samples=similar_samples,
            test_embedding_db=test_embedding_db
        )
    calib_no = 1 - calib_yes
    calib_entropy = calculate_entropy(np.array([calib_yes, calib_no]))

    # Final REAL result
    final_label = 1 if calib_yes >= 0.5 else 0
    final_result = {
        "e1": e1,
        "e2": e2,
        "raw_yes_prob": yes_prob,       # REAL LLM prob
        "raw_no_prob": no_prob,
        "calib_yes_prob": calib_yes,    # Calibrated prob
        "calib_no_prob": calib_no,
        "raw_entropy": raw_entropy,
        "calib_entropy": calib_entropy,
        "calibration_method": CALIBRATION_METHOD,
        "similar_samples": similar_samples,
        "performance_stats": performance_stats,
        "final_label": final_label,
        "final_judgment": "Has direct causal relationship" if final_label == 1 else "No direct causal relationship"
    }

    print(f"\n✅ FINAL REAL Result (Calibrated with {CALIBRATION_METHOD}):")
    print(f"  Calibrated Probability: Yes={calib_yes:.4f}, No={calib_no:.4f} (Entropy: {calib_entropy:.4f})")
    print(f"  Final Judgment: {final_result['final_judgment']}")
    print("="*80)
    return final_result

# ========== Test with REAL data (no fake) ==========
if __name__ == "__main__":
    # 测试示例（可替换为任意因果对）
    predict_new_sample_semantic("Playing computer games", "Eye strain")
    # 卸载LLM
    llm_client.unload_model()