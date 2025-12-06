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
