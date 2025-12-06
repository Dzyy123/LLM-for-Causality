from llm_utils import LocalLLMClient, ChatMessage, setup_logging

# ============== Configuration ==============
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"  # HuggingFace model identifier
LOCAL_PATH = "./models/qwen3-4b-thinking"  # Local storage path
MIRROR_URL = "https://hf-mirror.com"  # Use mirror for faster download in China (set to None for official HF)
DEVICE = "auto"  # "auto", "cuda", or "cpu"

# ============== Setup ==============
setup_logging(level="INFO")

CLIENT = LocalLLMClient(
    model_id=MODEL_ID,
    local_path=LOCAL_PATH,
    mirror_url=MIRROR_URL,
    device=DEVICE,
    max_tokens=4000
)

# TEST 1: Smoking -> Lung Cancer

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and lung cancer?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
print(f"Full response length: {len(response.content)} chars")

probs = CLIENT.get_token_probabilities(response, "mean", skip_thinking=True)
print(f"Token probabilities (mean) for yes: {probs.get('yes', 0.0)}")
print(f"Token probabilities (mean) for no: {probs.get('no', 0.0)}")

# TEST 2: Smoking -> Malaria

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and malaria?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
# print(f"Response: {response.content}")

probs = CLIENT.get_token_probabilities(response, "mean", skip_thinking=True)

print(f"Token probabilities (mean) for yes: {probs.get('yes', 0.0)}")
print(f"Token probabilities (mean) for no: {probs.get('no', 0.0)}")

# TEST 2: Smoking -> Malaria with Check

conv = CLIENT.start_conversation(
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only.",
    max_history=10  # Keep last 10 messages
)

response = conv.send("Is there any causal relationship between smoking and malaria?")
print(f"Turn 1: {response.crop_thinking().content}")

response = conv.send("Are you sure about your response? Answer yes if you are sure, no otherwise.")
print(f"Turn 2: {response.crop_thinking().content}")

probs = CLIENT.get_token_probabilities(response, "mean", skip_thinking=True)

print(f"Token probabilities (mean) for yes: {probs.get('yes', 0.0)}")
print(f"Token probabilities (mean) for no: {probs.get('no', 0.0)}")

CLIENT.unload_model()