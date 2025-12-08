"""
Local Model Conversation Example

Demonstrates downloading and using a local HuggingFace model with llm_utils.
"""

from llm_utils import LocalLLMClient, setup_logging

# ============== Configuration ==============
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"  # HuggingFace model identifier
LOCAL_PATH = "./models/qwen3-4b-thinking"  # Local storage path
MIRROR_URL = "https://hf-mirror.com"  # Use mirror for faster download in China (set to None for official HF)
DEVICE = "auto"  # "auto", "cuda", or "cpu"

# ============== Setup ==============
setup_logging(level="INFO")


def main():
    # Create client - automatically downloads if model not present locally
    print(f"Loading model: {MODEL_ID}")
    client = LocalLLMClient(
        model_id=MODEL_ID,
        local_path=LOCAL_PATH,
        mirror_url=MIRROR_URL,
        device=DEVICE,
        max_tokens=4000,
    )
    
    # Simple chat
    print("\n--- Simple Chat ---")
    response = client.chat("What is 2 + 2? Answer briefly.")
    print(f"Response: {response.content}")
    
    # Chat with system prompt
    print("\n--- Chat with System Prompt ---")
    response = client.chat(
        prompt="Explain quantum computing in one sentence.",
        system_prompt="You are a physics teacher. Keep explanations simple."
    )
    print(f"Response: {response.content}")
    
    # Get token probabilities (useful for causality judgment)
    print("\n--- Token Probabilities ---")
    response = client.chat(
        "Is the sky blue? Answer yes or no:",
        return_token_probs=True  # Enable probability tracking
    )
    print(f"Response: {response.content}")
    
    # Method 1: Skip thinking tokens directly
    probs = client.get_token_probabilities(response, skip_thinking=True)
    print(f"Token probabilities (skip_thinking=True): {probs}")
    
    # Method 2: Crop thinking first, then get probabilities
    cropped = response.crop_thinking()
    print(f"Cropped response: {cropped.content}")
    probs = client.get_token_probabilities(cropped)
    print(f"Token probabilities (after crop): {probs}")
    
    # Get token distribution at each position
    print("\n--- Token Distribution (Top-K) ---")
    dist = client.get_token_distributions(response, top_k=5, skip_thinking=True)
    print(f"Number of positions: {len(dist)}")
    if len(dist) > 0:
        print(f"Position 0 (first token) top-5: {dist[0]}")
    if len(dist) > 1:
        print(f"Position 1 (second token) top-5: {dist[1]}")
    if len(dist) > 2:
        print(f"Position 2 (third token) top-5: {dist[2]}")
    
    # Get token distribution with zero filtering
    print("\n--- Token Distribution (Skip Zeros) ---")
    dist_filtered = client.get_token_distributions(
        response, 
        top_k=10, 
        skip_zeros=True, 
        zero_threshold=0.001,  # Skip tokens with prob < 0.001
        skip_thinking=True
    )
    print(f"Number of positions: {len(dist_filtered)}")
    if len(dist_filtered) > 0:
        print(f"Position 0 (filtered, prob >= 0.001): {dist_filtered[0]}")
        print(f"  Number of tokens: {len(dist_filtered[0])}")
    
    # Analyze token distribution for causality judgment
    print("\n--- Token Distribution for Causality ---")
    response = client.chat(
        "Is there a causal relationship between smoking and lung cancer? Answer yes or no.",
        system_prompt="You are a medical expert. Answer concisely.",
        return_token_probs=True
    )
    cropped = response.crop_thinking()
    print(f"Answer: {cropped.content}")
    
    # Get distribution without filtering
    dist_all = client.get_token_distributions(cropped, top_k=10)
    if len(dist_all) > 0:
        print(f"\nFirst token distribution (top-10, all tokens):")
        for token, prob in list(dist_all[0].items())[:10]:
            print(f"  '{token}': {prob:.4f}")
    
    # Get distribution with zero filtering
    dist_significant = client.get_token_distributions(
        cropped, top_k=10, skip_zeros=True, zero_threshold=0.01
    )
    if len(dist_significant) > 0:
        print(f"\nFirst token distribution (top-10, prob >= 0.01):")
        for token, prob in list(dist_significant[0].items()):
            print(f"  '{token}': {prob:.4f}")

    
    # Multi-turn conversation (manual history management)
    print("\n--- Multi-turn Conversation (Manual) ---")
    from llm_utils import ChatMessage
    messages = [
        ChatMessage(role="system", content="You are a helpful math tutor."),
        ChatMessage(role="user", content="What is 15% of 200?"),
        ChatMessage(role="assistant", content="15% of 200 is 30."),
        ChatMessage(role="user", content="How did you calculate that?"),
    ]
    response = client.chat_with_history(messages)
    print(f"Response: {response.content}")
    
    # Multi-turn conversation (automatic history management)
    print("\n--- Multi-turn Conversation (Automatic) ---")
    conv = client.start_conversation(
        system_prompt="You are a helpful math tutor.",
        max_history=10  # Keep last 10 messages
    )
    
    response = conv.send("What is 20% of 150?")
    print(f"Turn 1: {response.content}")
    
    response = conv.send("What about 25% of the same number?")
    print(f"Turn 2: {response.content}")
    
    response = conv.send("Which one is larger?")
    print(f"Turn 3: {response.content}")
    
    print(f"Conversation has {conv.get_message_count()} messages")
    
    # Clean up
    client.unload_model()
    print("\nModel unloaded.")


if __name__ == "__main__":
    main()
