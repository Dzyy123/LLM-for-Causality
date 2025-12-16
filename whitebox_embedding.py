import numpy as np
from sentence_transformers import SentenceTransformer
import os
import warnings
warnings.filterwarnings("ignore")

def init_embedding_model(language: str = "en", local_model_path: str = None) -> SentenceTransformer:
    """Load local Embedding model (only local files)"""
    if local_model_path and os.path.exists(local_model_path):
        model = SentenceTransformer(local_model_path, local_files_only=True)
        print(f"✅ Loaded local Embedding model: {local_model_path}")
        return model
    
    preset_local_paths = {
        "en": "./models/all-MiniLM-L6-v2",
        "zh": "./models/text2vec-base-chinese"
    }
    target_path = preset_local_paths.get(language, "./models/all-MiniLM-L6-v2")
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"❌ Local Embedding model path not found: {target_path}\n"
            f"Download from https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2 first!"
        )
    
    model = SentenceTransformer(target_path, local_files_only=True)
    print(f"✅ Loaded local {language} Embedding model: {target_path}")
    return model

def generate_embedding(e1: str, e2: str, model: SentenceTransformer) -> np.ndarray:
    """Generate semantic vector for causal pair (e1→e2)"""
    causal_text = f"Causal inference: Does {e1} directly cause {e2}?"
    vec = model.encode(causal_text)
    vec = vec / np.linalg.norm(vec)  # Normalization
    return vec

def build_test_embedding_db(
    json_data: list, 
    model: SentenceTransformer, 
    save_path: str = "test_embedding_db.npz",
    pred_probs: list = None  
):
    """Build test set embedding DB (only REAL data)"""
    if pred_probs is None:
        raise ValueError("❌ pred_probs cannot be None! Use saved real probabilities.")
    
    embeddings = []       
    sample_ids = []       
    sample_texts = []     
    sample_performances = []  
    sample_pred_probs = []  
    sample_true_labels = [] 
    sample_errors = []      

    from whitebox_calibration import identify_extreme_error_samples

    # Extract real true labels
    true_labels = [item["causal"] for item in json_data]
    
    # Calculate REAL error (true - predicted)
    for p, y in zip(pred_probs, true_labels):
        error = y - p  # Real error
        sample_pred_probs.append(p)
        sample_true_labels.append(y)
        sample_errors.append(error)
    
    # Identify extreme samples
    extreme_result = identify_extreme_error_samples(pred_probs, true_labels)
    extreme_error_indices = [s["index"] for s in extreme_result["extreme_error_samples"]]
    extreme_correct_indices = [s["index"] for s in extreme_result["extreme_correct_samples"]]

    # Generate real semantic vectors
    for idx, item in enumerate(json_data):
        e1 = item["element1"]
        e2 = item["element2"]
        vec = generate_embedding(e1, e2, model)
        embeddings.append(vec)
        sample_ids.append(idx)
        sample_texts.append(f"{e1}→{e2}")
        
        # Mark performance
        if idx in extreme_error_indices:
            sample_performances.append(0)
        elif idx in extreme_correct_indices:
            sample_performances.append(1)
        else:
            sample_performances.append(2)

    # Save REAL data
    np.savez(
        save_path,
        embeddings=np.array(embeddings),
        sample_ids=sample_ids,
        sample_texts=sample_texts,
        sample_performances=sample_performances,
        sample_pred_probs=np.array(sample_pred_probs),
        sample_true_labels=np.array(sample_true_labels),
        sample_errors=np.array(sample_errors)
    )
    print(f"✅ Test set embedding DB saved (REAL data): {save_path}")
    return {
        "embeddings": embeddings,
        "sample_ids": sample_ids,
        "sample_texts": sample_texts,
        "sample_performances": sample_performances,
        "sample_pred_probs": sample_pred_probs,
        "sample_true_labels": sample_true_labels,
        "sample_errors": sample_errors
    }

def load_test_embedding_db(load_path: str = "test_embedding_db.npz") -> dict:
    """Load saved test set embedding DB (skip regeneration)"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ Saved embedding DB not found: {load_path}\nRun build_embedding_db.py first!")
    data = np.load(load_path, allow_pickle=True)
    print(f"✅ Loaded saved embedding DB (REAL data): {load_path}")
    return {
        "embeddings": data["embeddings"],
        "sample_ids": data["sample_ids"],
        "sample_texts": data["sample_texts"],
        "sample_performances": data["sample_performances"],
        "sample_pred_probs": data["sample_pred_probs"],
        "sample_true_labels": data["sample_true_labels"],
        "sample_errors": data["sample_errors"]
    }