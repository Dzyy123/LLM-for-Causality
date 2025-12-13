"""
Conversation Management Example

Demonstrates the automatic conversation history tracking feature
that works with all LLM clients (online and local).
"""

from llm_utils import LocalLLMClient, OnlineLLMClient, setup_logging

# Setup logging
setup_logging(level="INFO")


def demo_local_conversation():
    """Demonstrate conversation management with local model."""
    print("=" * 60)
    print("LOCAL MODEL CONVERSATION DEMO")
    print("=" * 60)
    
    # Create local client
    client = LocalLLMClient(
        model_id="Qwen/Qwen3-4B-Thinking-2507",
        local_path="./models/qwen3-4b-thinking",
        mirror_url="https://hf-mirror.com",
        device="auto",
        max_tokens=150
    )
    
    # Start a conversation with system prompt
    conv = client.start_conversation(
        system_prompt="You are a helpful math tutor. Keep answers concise.",
        max_history=10  # Keep last 10 messages (+ system)
    )
    
    print("\n--- Turn 1 ---")
    response = conv.send("What is 15% of 200?")
    print(f"User: What is 15% of 200?")
    print(f"Assistant: {response.content}")
    
    print("\n--- Turn 2 (uses context) ---")
    response = conv.send("How did you calculate that?")
    print(f"User: How did you calculate that?")
    print(f"Assistant: {response.content}")
    
    print("\n--- Turn 3 (uses context) ---")
    response = conv.send("What would 20% of the same number be?")
    print(f"User: What would 20% of the same number be?")
    print(f"Assistant: {response.content}")
    
    # Inspect conversation
    print(f"\n--- Conversation Summary ---")
    print(f"Total messages: {conv.get_message_count()}")
    print(f"Estimated tokens: {conv.get_token_estimate()}")
    print(f"\nFull history:")
    for i, msg in enumerate(conv.get_history(), 1):
        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"  {i}. [{msg.role}]: {content_preview}")
    
    # Clean up
    client.unload_model()
    print("\n✓ Local conversation demo completed\n")


def demo_online_conversation():
    """Demonstrate conversation management with online API."""
    print("=" * 60)
    print("ONLINE API CONVERSATION DEMO")
    print("=" * 60)
    
    # Configuration
    API_KEY = "sk-..."  # Replace with your API key
    BASE_URL = "https://llmapi.paratera.com/v1/"
    MODEL = "Qwen/Qwen3-Next-80B-A3B-Thinking"
    
    # Create online client
    client = OnlineLLMClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL,
        max_tokens=150
    )
    
    # Start a conversation
    conv = client.start_conversation(
        system_prompt="You are a helpful coding assistant.",
        max_history=20  # Keep last 20 messages
    )
    
    print("\n--- Turn 1 ---")
    response = conv.send("What is a Python decorator?")
    print(f"User: What is a Python decorator?")
    print(f"Assistant: {response.content}")
    
    print("\n--- Turn 2 (uses context) ---")
    response = conv.send("Can you show me a simple example?")
    print(f"User: Can you show me a simple example?")
    print(f"Assistant: {response.content}")
    
    print("\n--- Turn 3 (uses context) ---")
    response = conv.send("What are common use cases?")
    print(f"User: What are common use cases?")
    print(f"Assistant: {response.content}")
    
    # Demonstrate undo
    print("\n--- Undo last exchange ---")
    conv.remove_last_exchange()
    print(f"Removed last exchange. Messages now: {conv.get_message_count()}")
    
    # Demonstrate manual message injection
    print("\n--- Manual context injection ---")
    conv.add_message("user", "Actually, I prefer async/await examples.")
    conv.add_message("assistant", "Sure, I'll focus on async/await patterns.")
    
    response = conv.send("Show me an async decorator example.")
    print(f"User: Show me an async decorator example.")
    print(f"Assistant: {response.content}")
    
    print(f"\n--- Final message count: {conv.get_message_count()} ---")
    print("\n✓ Online conversation demo completed\n")


def demo_conversation_features():
    """Demonstrate advanced conversation features."""
    print("=" * 60)
    print("ADVANCED CONVERSATION FEATURES")
    print("=" * 60)
    
    # Use a simple client for demo
    client = LocalLLMClient(
        model_id="Qwen/Qwen3-4B-Thinking-2507",
        local_path="./models/qwen3-4b-thinking",
        mirror_url="https://hf-mirror.com",
        device="auto",
        max_tokens=50
    )
    
    # Feature 1: History limit
    print("\n--- Feature 1: Automatic History Trimming ---")
    conv = client.start_conversation(
        system_prompt="You are helpful.",
        max_history=4  # Keep only 4 messages total (including system)
    )
    
    conv.send("Message 1")
    conv.send("Message 2")
    conv.send("Message 3")
    print(f"After 3 exchanges: {conv.get_message_count()} messages")
    
    conv.send("Message 4")
    print(f"After 4th exchange (trimmed): {conv.get_message_count()} messages")
    print("Oldest messages were automatically removed!")
    
    # Feature 2: Clear history
    print("\n--- Feature 2: Clear History ---")
    conv.clear_history(keep_system=True)
    print(f"After clear (keep_system=True): {conv.get_message_count()} messages")
    
    # Feature 3: Manual history building
    print("\n--- Feature 3: Manual History Building ---")
    conv.add_message("user", "What is Python?")
    conv.add_message("assistant", "Python is a programming language.")
    conv.add_message("user", "Tell me more about it.")
    print(f"Manually built {conv.get_message_count()} messages")
    
    response = conv.send("What are its main features?")
    print(f"Response uses injected context: {response.content[:100]}...")
    
    # Feature 4: String representation
    print("\n--- Feature 4: Conversation Overview ---")
    print(conv)
    
    # Clean up
    client.unload_model()
    print("\n✓ Advanced features demo completed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CONVERSATION MANAGEMENT EXAMPLES")
    print("=" * 60 + "\n")
    
    # Demo with local model
    demo_local_conversation()
    
    # Demo with online API (uncomment and add API key to run)
    # demo_online_conversation()
    
    # Demo advanced features
    demo_conversation_features()
    
    print("=" * 60)
    print("ALL DEMOS COMPLETED")
    print("=" * 60)
