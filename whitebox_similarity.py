import numpy as np

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity (REAL semantic matching)"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def find_similar_test_samples(
    new_vec: np.ndarray,
    test_embedding_db: dict,
    top_k: int = 5,
    sim_threshold: float = 0.5  
) -> dict:
    """
    修复：恢复performance字段（校准函数依赖），同时保留预测概率+真实答案
    """
    test_embeddings = test_embedding_db["embeddings"]
    sample_ids = test_embedding_db["sample_ids"]
    sample_texts = test_embedding_db["sample_texts"]
    sample_performances = test_embedding_db["sample_performances"]  # 恢复performance字段
    sample_pred_probs = test_embedding_db["sample_pred_probs"]
    sample_true_labels = test_embedding_db["sample_true_labels"]

    # Calculate REAL similarity
    similarities = []
    for test_vec in test_embeddings:
        sim = calculate_cosine_similarity(new_vec, test_vec)
        similarities.append(sim)
    
    # Sort and filter
    sorted_indices = np.argsort(similarities)[::-1]
    similar_samples = []
    
    for idx in sorted_indices[:top_k]:
        sim = similarities[idx]
        if sim < sim_threshold:
            continue
        # 核心：恢复performance字段 + 预测概率 + 真实答案
        pred_prob = round(sample_pred_probs[idx], 4)
        true_label = "有因果" if sample_true_labels[idx] == 1 else "无因果"
        performance = sample_performances[idx]  # 恢复performance字段
        similar_samples.append({
            "sample_id": sample_ids[idx],
            "sample_text": sample_texts[idx],
            "similarity": sim,
            "performance": performance,  # 校准函数依赖的核心字段（必须保留）
            "model_pred_yes_prob": pred_prob,
            "true_answer": true_label
        })
    
    # 统计逻辑不变
    if not similar_samples:
        performance_stats = {
            "avg_similarity": 0.0
        }
    else:
        performance_stats = {
            "avg_similarity": round(np.mean([s["similarity"] for s in similar_samples]), 4)
        }

    return {
        "similar_samples": similar_samples,
        "performance_stats": performance_stats,
        "has_high_similarity": len(similar_samples) > 0 and performance_stats["avg_similarity"] >= 0.7
    }