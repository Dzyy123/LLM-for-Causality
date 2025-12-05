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
        max_tokens=200
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
    probs = client.get_token_probabilities(
        "Is the sky blue? Answer yes or no:",
        target_tokens=["yes", "no", "Yes", "No"]
    )
    print(f"Token probabilities: {probs}")
    
    # Multi-turn conversation
    print("\n--- Multi-turn Conversation ---")
    from llm_utils import ChatMessage
    messages = [
        ChatMessage(role="system", content="You are a helpful math tutor."),
        ChatMessage(role="user", content="What is 15% of 200?"),
        ChatMessage(role="assistant", content="15% of 200 is 30."),
        ChatMessage(role="user", content="How did you calculate that?"),
    ]
    response = client.chat_with_history(messages)
    print(f"Response: {response.content}")
    
    # Clean up
    client.unload_model()
    print("\nModel unloaded.")


if __name__ == "__main__":
    main()
