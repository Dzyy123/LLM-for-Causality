import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from whitebox_embedding import init_embedding_model, build_test_embedding_db
from whitebox_json_loader import load_causal_500_json, get_model_probs_for_json
from llm_utils import LocalLLMClient  # 保持你原有LLM客户端

# ========== Configuration (REAL paths only) ==========
TEST_JSON_PATH = "./causal_500.json"          
LOCAL_EMBEDDING_PATH = "./models/all-MiniLM-L6-v2"  
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 
QWEN_LOCAL_PATH = "./models/qwen3-4b-instruct" 
DEVICE = "cpu"                                
EMBEDDING_DB_SAVE_PATH = "test_embedding_db.npz"
PRED_PROBS_SAVE_PATH = "test_pred_probs.npz"   # 已保存的真实概率
EMBEDDING_LANGUAGE = "en"                     
BATCH_SIZE = 20                               

# ========== Save/Load REAL probabilities (skip regeneration) ==========
def save_pred_probs(pred_probs: list, save_path: str = PRED_PROBS_SAVE_PATH):
    """Save REAL test set probabilities (no fake)"""
    np.savez(save_path, pred_probs=np.array(pred_probs))
    print(f"✅ Saved REAL test set probabilities: {save_path}")

def load_pred_probs(load_path: str = PRED_PROBS_SAVE_PATH) -> list:
    """Load saved REAL probabilities (skip LLM prediction)"""
    if not os.path.exists(load_path):
        return None
    data = np.load(load_path, allow_pickle=True)
    pred_probs = data["pred_probs"].tolist()
    print(f"✅ Loaded saved REAL probabilities: {load_path}")
    return pred_probs

# ========== Core logic (only reuse saved data) ==========
def main():
    # 1. Validate REAL test set
    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(f"❌ REAL test set not found: {TEST_JSON_PATH}")
    
    # 2. Priority: load saved REAL probabilities (skip LLM)
    pred_probs = load_pred_probs()
    if pred_probs is not None:
        json_data = load_causal_500_json(TEST_JSON_PATH)
        if len(pred_probs) == len(json_data):
            print("ℹ️ Reuse saved REAL probabilities (skip LLM prediction)")
        else:
            raise ValueError(f"❌ Saved probabilities length mismatch! Regenerate REAL data (no fake).")
    else:
        # 3. Generate REAL probabilities (no fake)
        print("Step 1: Initialize LLM client for REAL prediction (no fake data)")
        try:
            llm_client = LocalLLMClient(
                model_id=QWEN_MODEL_ID,
                local_path=QWEN_LOCAL_PATH,
                device=DEVICE,
                max_tokens=4000
            )
        except Exception as e:
            raise RuntimeError(f"❌ LLM initialization failed: {e}\nFix LLM first (no fake data)!")
        
        # Load REAL test set
        print("\nStep 2: Load REAL test set data")
        json_data = load_causal_500_json(TEST_JSON_PATH)
        
        # Generate REAL probabilities
        print("\nStep 3: Generate REAL LLM probabilities (no fake)")
        pred_probs = get_model_probs_for_json(
            json_data=json_data,
            client=llm_client,
            batch_size=BATCH_SIZE
        )
        save_pred_probs(pred_probs)
        llm_client.unload_model()

    # 4. Reuse saved embedding DB (skip regeneration)
    if os.path.exists(EMBEDDING_DB_SAVE_PATH):
        print(f"\nℹ️ Reuse saved REAL embedding DB (skip generation): {EMBEDDING_DB_SAVE_PATH}")
    else:
        # 5. Build REAL embedding DB (no fake)
        print("\nStep 4: Initialize REAL Embedding model")
        embedding_model = init_embedding_model(
            language=EMBEDDING_LANGUAGE,
            local_model_path=LOCAL_EMBEDDING_PATH
        )
        
        print("\nStep 5: Build REAL embedding DB (no fake data)")
        build_test_embedding_db(
            json_data=json_data,
            model=embedding_model,
            save_path=EMBEDDING_DB_SAVE_PATH,
            pred_probs=pred_probs  # Only REAL probs
        )

    print(f"\n🎉 All completed (100% REAL data)!")
    print(f"  - Saved REAL probabilities: {PRED_PROBS_SAVE_PATH}")
    print(f"  - Saved REAL embedding DB: {EMBEDDING_DB_SAVE_PATH}")
    print(f"  - Test set samples: {len(json_data)}")

if __name__ == "__main__":
    main()