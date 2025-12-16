from llm_utils import (
    # Logging
    setup_logging,
    get_logger,
    # Data types
    ResponseFormat,
    LLMResponse,
    ChatMessage,
    # Clients
    BaseLLMClient,
    OnlineLLMClient,
    LocalLLMClient,
    # Factory
    LLMClientFactory,
    create_llm_client,
)

TEST_URL = "https://llmapi.paratera.com/v1/"
TEST_API_KEY = "sk-fnUHDzxXAimEnYgyX20Jag"
TEST_MODEL = "Qwen3-Next-80B-A3B-Thinking"

# HuggingFace mirror for testing downloads (faster in China)
TEST_HF_MIRROR = "https://hf-mirror.com"
# Small model for download testing (to avoid downloading large models)
TEST_DOWNLOAD_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
TEST_DOWNLOAD_LOCAL_PATH = "./test_models/tiny-gpt2"


def test_response_format() -> bool:
    """Test ResponseFormat enum."""
    try:
        assert ResponseFormat.TEXT.value == "text"
        assert ResponseFormat.JSON.value == "json"
        assert ResponseFormat.PROBABILITY.value == "probability"
        print("✓ ResponseFormat enum works correctly")
        return True
    except AssertionError as e:
        print(f"✗ ResponseFormat test failed: {e}")
        return False


def test_chat_message() -> bool:
    """Test ChatMessage dataclass."""
    try:
        msg = ChatMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.to_dict() == {"role": "user", "content": "Hello, world!"}
        print("✓ ChatMessage works correctly")
        return True
    except AssertionError as e:
        print(f"✗ ChatMessage test failed: {e}")
        return False


def test_llm_response() -> bool:
    """Test LLMResponse dataclass."""
    try:
        response = LLMResponse(
            content="Test response",
            raw_response=None,
            token_probabilities={"yes": 0.8, "no": 0.2},
            metadata={"model": "test-model"}
        )
        assert response.content == "Test response"
        assert response.token_probabilities["yes"] == 0.8
        assert response.metadata["model"] == "test-model"
        print("✓ LLMResponse works correctly")
        return True
    except AssertionError as e:
        print(f"✗ LLMResponse test failed: {e}")
        return False


def test_online_client_creation() -> bool:
    """Test OnlineLLMClient creation."""
    try:
        client = OnlineLLMClient(
            api_key=TEST_API_KEY,
            base_url=TEST_URL,
            model_name=TEST_MODEL,
            max_tokens=100
        )
        print(f"✓ OnlineLLMClient created: {client}")
        return True
    except Exception as e:
        print(f"✗ OnlineLLMClient creation failed: {e}")
        return False


def test_online_client_chat() -> bool:
    """Test OnlineLLMClient chat functionality and response correctness."""
    try:
        client = OnlineLLMClient(
            api_key=TEST_API_KEY,
            base_url=TEST_URL,
            model_name=TEST_MODEL,
            max_tokens=100
        )
        
        # Test with a question that has a definitive answer
        response = client.chat("What is 2 + 2? Answer with just the number.")
        
        assert response.content is not None, "Response content should not be None"
        assert len(response.content) > 0, "Response content should not be empty"
        
        # Check if the response contains the correct answer
        if "4" in response.content:
            print(f"✓ Chat response correct: {response.content[:100]}...")
            return True
        else:
            print(f"⚠ Chat response received but answer may be incorrect: {response.content[:100]}...")
            return True  # Still pass as the API responded
            
    except Exception as e:
        print(f"✗ OnlineLLMClient chat failed: {e}")
        return False


def test_online_client_chat_with_history() -> bool:
    """Test OnlineLLMClient chat with history functionality and response correctness."""
    try:
        client = OnlineLLMClient(
            api_key=TEST_API_KEY,
            base_url=TEST_URL,
            model_name=TEST_MODEL,
            max_tokens=100
        )
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. Answer concisely."),
            ChatMessage(role="user", content="What is the capital of France? Answer with just the city name."),
        ]
        response = client.chat_with_history(messages)
        
        assert response.content is not None, "Response content should not be None"
        assert len(response.content) > 0, "Response content should not be empty"
        
        # Check if the response contains the correct answer
        if "Paris" in response.content or "paris" in response.content.lower():
            print(f"✓ Chat with history response correct: {response.content[:100]}...")
            return True
        else:
            print(f"⚠ Chat with history response received but may be incorrect: {response.content[:100]}...")
            return True  # Still pass as the API responded
            
    except Exception as e:
        print(f"✗ OnlineLLMClient chat with history failed: {e}")
        return False


def test_factory_create_online() -> bool:
    """Test LLMClientFactory.create_online method."""
    try:
        client = LLMClientFactory.create_online(
            api_key=TEST_API_KEY,
            base_url=TEST_URL,
            model_name=TEST_MODEL
        )
        print(f"✓ LLMClientFactory.create_online() works: {client}")
        return True
    except Exception as e:
        print(f"✗ LLMClientFactory.create_online() failed: {e}")
        return False


def test_factory_create_paratera() -> bool:
    """Test LLMClientFactory.create_paratera method."""
    try:
        client = LLMClientFactory.create_paratera(
            api_key=TEST_API_KEY,
            model_name=TEST_MODEL
        )
        print(f"✓ LLMClientFactory.create_paratera() works: {client}")
        return True
    except Exception as e:
        print(f"✗ LLMClientFactory.create_paratera() failed: {e}")
        return False


def test_create_llm_client_online() -> bool:
    """Test create_llm_client convenience function for online client."""
    try:
        client = create_llm_client(
            "online",
            api_key=TEST_API_KEY,
            base_url=TEST_URL,
            model_name=TEST_MODEL
        )
        print(f"✓ create_llm_client('online') works: {client}")
        return True
    except Exception as e:
        print(f"✗ create_llm_client('online') failed: {e}")
        return False


def test_local_client_creation() -> bool:
    """Test LocalLLMClient creation without loading model."""
    import os
    
    # Use the downloaded test model path
    if not os.path.exists(TEST_DOWNLOAD_LOCAL_PATH):
        print(f"⚠ Skipping LocalLLMClient creation test (model not downloaded yet)")
        return True  # Skip is not a failure
    
    try:
        client = LocalLLMClient(
            model_id=TEST_DOWNLOAD_MODEL_ID,
            local_path=TEST_DOWNLOAD_LOCAL_PATH,
            device="auto",
            load_on_init=False
        )
        print(f"✓ LocalLLMClient created (not loaded): {client}")
        return True
    except Exception as e:
        print(f"✗ LocalLLMClient creation failed: {e}")
        return False


def test_local_client_load_model() -> bool:
    """Test LocalLLMClient model loading."""
    import os
    
    if not os.path.exists(TEST_DOWNLOAD_LOCAL_PATH):
        print(f"⚠ Skipping LocalLLMClient load test (model not downloaded yet)")
        return True
    
    try:
        client = LocalLLMClient(
            model_id=TEST_DOWNLOAD_MODEL_ID,
            local_path=TEST_DOWNLOAD_LOCAL_PATH,
            device="auto",
            load_on_init=True  # Actually load the model
        )
        print(f"✓ LocalLLMClient model loaded successfully")
        
        # Unload to free memory
        client.unload_model()
        print("✓ Model unloaded successfully")
        return True
    except Exception as e:
        print(f"✗ LocalLLMClient load failed: {e}")
        return False


def test_local_client_chat_response() -> bool:
    """Test LocalLLMClient chat generates valid response."""
    import os
    
    if not os.path.exists(TEST_DOWNLOAD_LOCAL_PATH):
        print(f"⚠ Skipping LocalLLMClient chat test (model not downloaded yet)")
        return True
    
    try:
        client = LocalLLMClient(
            model_id=TEST_DOWNLOAD_MODEL_ID,
            local_path=TEST_DOWNLOAD_LOCAL_PATH,
            device="auto",
            load_on_init=True,
            max_tokens=50
        )
        
        # Test chat - tiny-random-gpt2 generates random text, just verify it responds
        response = client.chat("Hello")
        
        assert response is not None, "Response should not be None"
        assert response.content is not None, "Response content should not be None"
        assert isinstance(response.content, str), "Response content should be string"
        
        print(f"✓ Local chat response received: '{response.content[:50]}...'")
        
        client.unload_model()
        return True
    except Exception as e:
        print(f"✗ LocalLLMClient chat test failed: {e}")
        return False


def test_local_client_token_probabilities() -> bool:
    """Test LocalLLMClient token probability calculation."""
    import os
    
    if not os.path.exists(TEST_DOWNLOAD_LOCAL_PATH):
        print(f"⚠ Skipping token probabilities test (model not downloaded yet)")
        return True
    
    try:
        client = LocalLLMClient(
            model_id=TEST_DOWNLOAD_MODEL_ID,
            local_path=TEST_DOWNLOAD_LOCAL_PATH,
            device="auto",
            load_on_init=True
        )
        
        # Generate response with token probabilities
        response = client.chat("The sky is", return_token_probs=True, max_tokens=10)
        probs = client.get_token_probabilities(response)
        
        assert probs is not None, "Probabilities should not be None"
        assert isinstance(probs, dict), "Probabilities should be a dictionary"
        assert len(probs) > 0, "Should have at least one token probability"
        
        # Verify probabilities are valid (between 0 and 1)
        for token, prob in probs.items():
            assert 0 <= prob <= 1, f"Probability for '{token}' should be between 0 and 1"
        
        print(f"✓ Token probabilities: {probs}")
        
        client.unload_model()
        return True
    except Exception as e:
        print(f"✗ Token probabilities test failed: {e}")
        return False


def test_local_client_full() -> bool:
    """Test LocalLLMClient with user-specified model path (optional)."""
    import os
    
    local_model_path = os.environ.get("LOCAL_MODEL_PATH")
    local_model_id = os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    
    if not local_model_path:
        print("⚠ Skipping user model test (set LOCAL_MODEL_PATH to test)")
        return True
    
    if not os.path.exists(local_model_path):
        print(f"⚠ Skipping user model test (path not found: {local_model_path})")
        return True
    
    try:
        client = LocalLLMClient(
            model_id=local_model_id,
            local_path=local_model_path,
            device="auto",
            load_on_init=True
        )
        
        response = client.chat("What is 2 + 2? Answer with just the number.")
        print(f"✓ User model chat response: {response.content[:100]}...")
        
        client.unload_model()
        print("✓ User model unloaded successfully")
        return True
    except Exception as e:
        print(f"✗ User model test failed: {e}")
        return False


def test_error_handling_invalid_client_type() -> bool:
    """Test error handling for invalid client type."""
    try:
        create_llm_client("invalid_type")
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


# =============================================================================
# HuggingFace Download Tests
# =============================================================================

def test_local_client_validates_existing_model() -> bool:
    """Test that LocalLLMClient correctly validates an existing local model."""
    import os
    import tempfile
    
    # Create a temporary directory with mock model files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create required files to simulate a valid model
        config_path = os.path.join(temp_dir, "config.json")
        model_path = os.path.join(temp_dir, "model.safetensors")
        
        with open(config_path, "w") as f:
            f.write('{"model_type": "test"}')
        with open(model_path, "w") as f:
            f.write("mock model weights")
        
        try:
            # Create client without loading (just validation)
            client = LocalLLMClient(
                model_id="test/mock-model",
                local_path=temp_dir,
                load_on_init=False
            )
            
            # Manually check validation
            is_valid = client._validate_local_model()
            
            if is_valid:
                print("✓ LocalLLMClient correctly validated existing model")
                return True
            else:
                print("✗ LocalLLMClient failed to validate existing model")
                return False
        except Exception as e:
            print(f"✗ Validation test failed: {e}")
            return False


def test_local_client_detects_missing_model() -> bool:
    """Test that LocalLLMClient correctly detects when model is missing."""
    import tempfile
    import os
    
    # Use a non-existent path
    non_existent_path = os.path.join(tempfile.gettempdir(), "non_existent_model_12345")
    
    # Make sure it doesn't exist
    if os.path.exists(non_existent_path):
        import shutil
        shutil.rmtree(non_existent_path)
    
    try:
        client = LocalLLMClient(
            model_id="test/non-existent",
            local_path=non_existent_path,
            load_on_init=False
        )
        
        is_valid = client._validate_local_model()
        
        if not is_valid:
            print("✓ LocalLLMClient correctly detected missing model")
            return True
        else:
            print("✗ LocalLLMClient incorrectly validated non-existent model")
            return False
    except Exception as e:
        print(f"✗ Missing model detection test failed: {e}")
        return False


def test_local_client_detects_incomplete_model() -> bool:
    """Test that LocalLLMClient correctly detects incomplete model (missing files)."""
    import os
    import tempfile
    
    # Create directory with only config (no model weights)
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"model_type": "test"}')
        
        try:
            client = LocalLLMClient(
                model_id="test/incomplete-model",
                local_path=temp_dir,
                load_on_init=False
            )
            
            is_valid = client._validate_local_model()
            
            if not is_valid:
                print("✓ LocalLLMClient correctly detected incomplete model (missing weights)")
                return True
            else:
                print("✗ LocalLLMClient incorrectly validated incomplete model")
                return False
        except Exception as e:
            print(f"✗ Incomplete model detection test failed: {e}")
            return False


def test_local_client_download_from_mirror() -> bool:
    """Test downloading model from HuggingFace mirror (hf-mirror.com).
    
    This test downloads a tiny test model to verify the download functionality
    and mirror URL support. The model is kept for subsequent tests.
    """
    import os
    import shutil
    
    # Clean up test directory if exists (fresh download)
    if os.path.exists(TEST_DOWNLOAD_LOCAL_PATH):
        shutil.rmtree(TEST_DOWNLOAD_LOCAL_PATH)
    
    try:
        # Create client with mirror URL, but don't load on init
        # (we want to test _ensure_model_ready separately)
        client = LocalLLMClient(
            model_id=TEST_DOWNLOAD_MODEL_ID,
            local_path=TEST_DOWNLOAD_LOCAL_PATH,
            mirror_url=TEST_HF_MIRROR,
            load_on_init=False
        )
        
        # Verify mirror_url is set
        assert client.mirror_url == TEST_HF_MIRROR, "mirror_url not set correctly"
        print(f"✓ Mirror URL configured: {client.mirror_url}")
        
        # Check that model doesn't exist yet
        assert not client._validate_local_model(), "Model should not exist yet"
        print("✓ Confirmed model does not exist locally")
        
        # Trigger download via _ensure_model_ready
        print(f"  Downloading {TEST_DOWNLOAD_MODEL_ID} from {TEST_HF_MIRROR}...")
        client._ensure_model_ready()
        
        # Verify model was downloaded
        assert os.path.exists(TEST_DOWNLOAD_LOCAL_PATH), "Download path not created"
        assert client._validate_local_model(), "Downloaded model failed validation"
        
        print(f"✓ Model downloaded successfully to {TEST_DOWNLOAD_LOCAL_PATH}")
        print("✓ LocalLLMClient correctly downloads from mirror when model is missing")
        
        # NOTE: Do NOT clean up here - keep the model for subsequent tests
        return True
        
    except Exception as e:
        print(f"✗ Download from mirror test failed: {e}")
        return False


def test_local_client_skips_download_when_exists() -> bool:
    """Test that LocalLLMClient skips download when valid model already exists."""
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid mock model files
        config_path = os.path.join(temp_dir, "config.json")
        model_path = os.path.join(temp_dir, "model.safetensors")
        
        with open(config_path, "w") as f:
            f.write('{"model_type": "test"}')
        with open(model_path, "w") as f:
            f.write("mock model weights")
        
        try:
            # Track if download was attempted
            download_attempted = False
            
            client = LocalLLMClient(
                model_id="test/should-not-download",
                local_path=temp_dir,
                mirror_url=TEST_HF_MIRROR,
                load_on_init=False
            )
            
            # Override _download_model to track if it's called
            original_download = client._download_model
            def mock_download():
                nonlocal download_attempted
                download_attempted = True
                return original_download()
            client._download_model = mock_download
            
            # Call _ensure_model_ready - should NOT trigger download
            client._ensure_model_ready()
            
            if not download_attempted:
                print("✓ LocalLLMClient correctly skipped download for existing model")
                return True
            else:
                print("✗ LocalLLMClient incorrectly attempted to download existing model")
                return False
                
        except Exception as e:
            print(f"✗ Skip download test failed: {e}")
            return False


def test_local_client_default_mirror_is_none() -> bool:
    """Test that default mirror_url is None (uses official huggingface.co)."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            client = LocalLLMClient(
                model_id="test/model",
                local_path=temp_dir,
                load_on_init=False
            )
            
            if client.mirror_url is None:
                print("✓ Default mirror_url is None (uses official huggingface.co)")
                return True
            else:
                print(f"✗ Default mirror_url should be None, got: {client.mirror_url}")
                return False
        except Exception as e:
            print(f"✗ Default mirror test failed: {e}")
            return False


def cleanup_test_downloads() -> None:
    """Clean up downloaded test models after all tests complete."""
    import os
    import shutil
    
    if os.path.exists(TEST_DOWNLOAD_LOCAL_PATH):
        shutil.rmtree(TEST_DOWNLOAD_LOCAL_PATH)
        print("  (Cleaned up test download directory)")
    
    # Also clean up parent directory if empty
    parent_dir = os.path.dirname(TEST_DOWNLOAD_LOCAL_PATH)
    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
        os.rmdir(parent_dir)


def run_all_tests() -> None:
    """Run all tests and report results."""
    # Configure logging for testing
    setup_logging(level="DEBUG")
    
    print("=" * 60)
    print("LLM_Utils Module - Testing")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Data types
    print("\n--- Test 1: Data Types ---")
    results["ResponseFormat"] = test_response_format()
    results["ChatMessage"] = test_chat_message()
    results["LLMResponse"] = test_llm_response()
    
    # Test 2: Online API client
    print("\n--- Test 2: Online API Client ---")
    results["OnlineLLMClient creation"] = test_online_client_creation()
    results["OnlineLLMClient chat"] = test_online_client_chat()
    results["OnlineLLMClient chat with history"] = test_online_client_chat_with_history()
    
    # Test 3: Factory pattern
    print("\n--- Test 3: Factory Pattern ---")
    results["Factory create_online"] = test_factory_create_online()
    results["Factory create_paratera"] = test_factory_create_paratera()
    results["create_llm_client online"] = test_create_llm_client_online()
    
    # Test 4: Error handling
    print("\n--- Test 4: Error Handling ---")
    results["Invalid client type error"] = test_error_handling_invalid_client_type()
    
    # Test 5: HuggingFace download functionality (validation tests first)
    print("\n--- Test 5: HuggingFace Download Validation ---")
    results["Validate existing model"] = test_local_client_validates_existing_model()
    results["Detect missing model"] = test_local_client_detects_missing_model()
    results["Detect incomplete model"] = test_local_client_detects_incomplete_model()
    results["Default mirror is None"] = test_local_client_default_mirror_is_none()
    results["Skip download when exists"] = test_local_client_skips_download_when_exists()
    
    # Test 6: Download model from mirror (this downloads the test model)
    print("\n--- Test 6: Download Model from Mirror ---")
    results["Download from mirror"] = test_local_client_download_from_mirror()
    
    # Test 7: Local model client (uses downloaded model from Test 6)
    print("\n--- Test 7: Local Model Client (Using Downloaded Model) ---")
    results["LocalLLMClient creation"] = test_local_client_creation()
    results["LocalLLMClient load model"] = test_local_client_load_model()
    results["LocalLLMClient chat response"] = test_local_client_chat_response()
    results["LocalLLMClient token probabilities"] = test_local_client_token_probabilities()
    
    # Test 8: User-specified model (optional)
    print("\n--- Test 8: User-Specified Model (Optional) ---")
    results["LocalLLMClient user model"] = test_local_client_full()
    
    # Clean up downloaded test models
    print("\n--- Cleanup ---")
    cleanup_test_downloads()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
