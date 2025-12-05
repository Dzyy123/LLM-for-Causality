"""
Online LLM client for API-based models.

This module provides a client for interacting with online LLM APIs
such as OpenAI, Anthropic Claude, Google Gemini, and other compatible APIs.
"""

from typing import Dict, List, Optional

from llm_utils.logging_config import get_logger
from llm_utils.base_client import BaseLLMClient
from llm_utils.data_types import LLMResponse, ChatMessage


# Module-level logger
logger = get_logger("online_client")


class OnlineLLMClient(BaseLLMClient):
    """Client for online LLM APIs (OpenAI-compatible, Gemini, Claude, etc.).
    
    This class provides a unified interface for various online LLM APIs
    that follow the OpenAI-compatible chat completions format.
    
    Attributes:
        api_key (str): API key for authentication.
        base_url (str): Base URL for the API endpoint.
        api_type (str): Type of API ('openai', 'anthropic', 'google', 'paratera').
    
    Example:
        >>> # OpenAI-compatible API
        >>> client = OnlineLLMClient(
        ...     api_key="sk-xxx",
        ...     base_url="https://api.openai.com/v1/",
        ...     model_name="gpt-4"
        ... )
        
        >>> # Claude API (Anthropic)
        >>> client = OnlineLLMClient(
        ...     api_key="sk-ant-xxx",
        ...     base_url="https://api.anthropic.com/v1/",
        ...     model_name="claude-3-opus-20240229",
        ...     api_type="anthropic"
        ... )
    """
    
    # Predefined API configurations
    API_CONFIGS = {
        "openai": {
            "base_url": "https://api.openai.com/v1/",
            "default_model": "gpt-4"
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1/",
            "default_model": "claude-3-opus-20240229"
        },
        "google": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/",
            "default_model": "gemini-pro"
        },
        "paratera": {
            "base_url": "https://llmapi.paratera.com/v1/",
            "default_model": "Qwen3-Next-80B-A3B-Thinking"
        }
    }
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        api_type: str = "openai"
    ) -> None:
        """Initialize the online LLM client.
        
        Args:
            api_key (str): API key for authentication.
            base_url (Optional[str]): Base URL for the API endpoint. If None,
                uses default for the specified api_type.
            model_name (Optional[str]): Name of the model. If None, uses
                default for api_type.
            temperature (float): Sampling temperature for generation.
                Defaults to 0.7.
            max_tokens (int): Maximum number of tokens in response.
                Defaults to 200.
            api_type (str): Type of API backend ('openai', 'anthropic',
                'google', 'paratera'). Defaults to 'openai'.
        """
        # Get default configuration for API type
        config = self.API_CONFIGS.get(api_type, self.API_CONFIGS["openai"])
        
        actual_base_url = base_url or config["base_url"]
        actual_model_name = model_name or config["default_model"]
        
        super().__init__(actual_model_name, temperature, max_tokens)
        
        self.api_key = api_key
        self.base_url = actual_base_url
        self.api_type = api_type
        self._client = None
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the API client based on API type.
        
        Raises:
            ImportError: If required package is not installed.
        """
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self._is_initialized = True
            logger.info("Online LLM client initialized: %s", self.model_name)
        except ImportError as e:
            raise ImportError(
                "openai package is required for OnlineLLMClient. "
                "Install with: pip install openai"
            ) from e
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat message and get a response.
        
        Args:
            prompt (str): The user's input prompt.
            system_prompt (Optional[str]): Optional system prompt to set context.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content.strip()
        
        return LLMResponse(
            content=content,
            raw_response=response,
            metadata={
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", None
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", None)
                }
            }
        )
    
    def chat_with_history(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """Send a conversation with history and get a response.
        
        Args:
            messages (List[ChatMessage]): List of chat messages representing
                the conversation.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        formatted_messages = [msg.to_dict() for msg in messages]
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.choices[0].message.content.strip()
        
        return LLMResponse(
            content=content,
            raw_response=response,
            metadata={
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", None
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", None)
                }
            }
        )
    
    def get_token_probabilities(
        self,
        prompt: str,
        target_tokens: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get token probabilities (limited support for online APIs).
        
        Note:
            Most online APIs do not provide direct token probability access.
            This method returns empty dict for unsupported APIs.
        
        Args:
            prompt (str): The input prompt.
            target_tokens (Optional[List[str]]): Optional list of tokens to get
                probabilities for.
        
        Returns:
            Dict[str, float]: Dictionary mapping tokens to their probabilities.
        """
        # Most online APIs don't support logprobs directly
        # For APIs that do support it, we would need to enable logprobs
        logger.warning(
            "Token probabilities are not directly available "
            "for most online APIs."
        )
        return {}
