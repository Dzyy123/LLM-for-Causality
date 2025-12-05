"""
Online Model Conversation Example

Demonstrates using online LLM APIs with llm_utils.
"""

from llm_utils import OnlineLLMClient, ChatMessage, setup_logging

# ============== Configuration ==============
API_KEY = "sk-fnUHDzxXAimEnYgyX20Jag"  # Your API key
BASE_URL = "https://llmapi.paratera.com/v1/"  # API endpoint
MODEL_NAME = "Qwen3-Next-80B-A3B-Thinking"  # Model to use

# ============== Setup ==============
setup_logging(level="INFO")


def main():
    # Create online client
    client = OnlineLLMClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        max_tokens=200
    )
    print(f"Connected to: {MODEL_NAME}")
    
    # Simple chat
    print("\n--- Simple Chat ---")
    response = client.chat("What is the capital of France?")
    print(f"Response: {response.content}")
    
    # Chat with system prompt
    print("\n--- Chat with System Prompt ---")
    response = client.chat(
        prompt="What causes rain?",
        system_prompt="You are a meteorologist. Be concise."
    )
    print(f"Response: {response.content}")
    
    # Multi-turn conversation
    print("\n--- Multi-turn Conversation ---")
    messages = [
        ChatMessage(role="system", content="You are a helpful math tutor."),
        ChatMessage(role="user", content="What is 15% of 200?"),
        ChatMessage(role="assistant", content="15% of 200 is 30."),
        ChatMessage(role="user", content="How did you calculate that?"),
    ]
    response = client.chat_with_history(messages)
    print(f"Response: {response.content}")
    
    # Access response metadata
    print(f"\nToken usage: {response.metadata.get('usage', 'N/A')}")


if __name__ == "__main__":
    main()
