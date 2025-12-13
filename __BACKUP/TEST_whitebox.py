from llm_utils import LocalLLMClient, ChatMessage, setup_logging

# ============== Configuration ==============
# MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"  # HuggingFace model identifier
# LOCAL_PATH = "./models/qwen3-4b-thinking"  # Local storage path
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # HuggingFace model identifier
LOCAL_PATH = "./models/qwen3-4b-instruct"  # Local storage path
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
print(f"Response: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

# TEST 2: Smoking -> Malaria

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and malaria?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
print(f"Response: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

# TEST 3: Smoking -> Pulmonary Edema

response = CLIENT.chat(
    prompt="Is there any causal relationship between smoking and pulmonary edema?",
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, answer yes or no only."
)
print(f"Response: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

# TEST 4: Smoking -> Pulmonary Edema with Check, multiple selection choice

conv = CLIENT.start_conversation(
    system_prompt="You are a professional medical assistant. After detailed thinking and deductions, according to your confidence in the answer, answer **in ONLY one letter**: \"A\" for absolutely yes, \"B\" for probably yes, \"C\" for probably no, and \"D\" for absolutely no.",
    max_history=10  # Keep last 10 messages
)

response = conv.send("Is there any causal relationship between smoking and pulmonary edema?")
print(f"Turn 1: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

response = conv.send("I think the answer is yes. Answer me, are you firmly sure about your previous answer?")
print(f"Turn 2: {response.crop_thinking().content}")

probs = CLIENT.get_token_distributions(response, skip_zeros=True, skip_thinking=True)
print(f"Token distributions:\n{probs}")

CLIENT.unload_model()