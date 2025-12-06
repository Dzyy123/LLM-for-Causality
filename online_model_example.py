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
    
    # Multi-turn conversation (manual history management)
    print("\n--- Multi-turn Conversation (Manual) ---")
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
        system_prompt="You are a helpful coding assistant.",
        max_history=20  # Keep last 20 messages
    )
    
    response = conv.send("What is a Python list?")
    print(f"Turn 1: {response.content[:100]}...")
    
    response = conv.send("How do I append to it?")
    print(f"Turn 2: {response.content[:100]}...")
    
    response = conv.send("Show me an example.")
    print(f"Turn 3: {response.content[:100]}...")
    
    print(f"\nConversation has {conv.get_message_count()} messages")
    print(f"Estimated tokens: {conv.get_token_estimate()}")
    
    # Access response metadata
    print(f"\nToken usage: {response.metadata.get('usage', 'N/A')}")


if __name__ == "__main__":
    main()
