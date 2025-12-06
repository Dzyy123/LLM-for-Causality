"""
LLM Utilities Package for Causal Inference Framework.

This package provides unified interfaces for both local and online LLM models,
supporting various APIs (OpenAI-compatible, Gemini, Claude, etc.) and local
HuggingFace models with automatic downloading and validation.

Example:
    >>> # Using online API
    >>> from llm_utils import OnlineLLMClient
    >>> client = OnlineLLMClient(
    ...     api_key="your-api-key",
    ...     base_url="https://api.openai.com/v1/",
    ...     model_name="gpt-4"
    ... )
    >>> response = client.chat("What is causality?")
    
    >>> # Using local model
    >>> from llm_utils import LocalLLMClient
    >>> local_model = LocalLLMClient(
    ...     model_id="Qwen/Qwen2.5-7B-Instruct",
    ...     local_path="./models/qwen"
    ... )
    >>> response = local_model.chat("What is causality?")
    
    >>> # Using factory
    >>> from llm_utils import create_llm_client, LLMClientFactory
    >>> client = create_llm_client("online", api_key="sk-xxx")
    
    >>> # Configure logging (optional, auto-configured with INFO level)
    >>> from llm_utils import setup_logging
    >>> setup_logging(level="DEBUG")  # Enable debug output

:author: LLM-for-Causality Team
:date: 2025
"""

# Import logging configuration first to ensure logging is set up
from llm_utils.logging_config import setup_logging, get_logger

from llm_utils.data_types import (
    ResponseFormat,
    LLMResponse,
    ChatMessage,
)

from llm_utils.conversation import Conversation

from llm_utils.base_client import BaseLLMClient

from llm_utils.online_client import OnlineLLMClient

from llm_utils.local_client import LocalLLMClient

from llm_utils.factory import (
    LLMClientFactory,
    create_llm_client,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Data types
    "ResponseFormat",
    "LLMResponse",
    "ChatMessage",
    # Conversation
    "Conversation",
    # Clients
    "BaseLLMClient",
    "OnlineLLMClient",
    "LocalLLMClient",
    # Factory
    "LLMClientFactory",
    "create_llm_client",
]
