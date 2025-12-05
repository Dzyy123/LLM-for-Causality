"""
Factory classes and convenience functions for creating LLM clients.

This module provides factory patterns and helper functions to simplify
the creation of LLM client instances.
"""

from typing import Optional

from llm_utils.base_client import BaseLLMClient
from llm_utils.online_client import OnlineLLMClient
from llm_utils.local_client import LocalLLMClient


class LLMClientFactory:
    """Factory class for creating LLM clients.
    
    This class provides convenient methods to create LLM clients
    for various backends with minimal configuration.
    
    Example:
        >>> # Create online client
        >>> client = LLMClientFactory.create_online(
        ...     api_key="sk-xxx",
        ...     api_type="openai",
        ...     model_name="gpt-4"
        ... )
        
        >>> # Create local client
        >>> client = LLMClientFactory.create_local(
        ...     model_id="Qwen/Qwen2.5-7B-Instruct",
        ...     local_path="./models/qwen"
        ... )
    """
    
    @staticmethod
    def create_online(
        api_key: str,
        api_type: str = "openai",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> OnlineLLMClient:
        """Create an online LLM client.
        
        Args:
            api_key (str): API key for authentication.
            api_type (str): Type of API ('openai', 'anthropic', 'google',
                'paratera'). Defaults to 'openai'.
            model_name (Optional[str]): Model name (uses default if None).
            base_url (Optional[str]): Custom base URL (uses default if None).
            **kwargs: Additional parameters passed to OnlineLLMClient.
        
        Returns:
            OnlineLLMClient: Configured OnlineLLMClient instance.
        """
        return OnlineLLMClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            api_type=api_type,
            **kwargs
        )
    
    @staticmethod
    def create_local(
        model_id: str,
        local_path: str,
        device: str = "auto",
        **kwargs
    ) -> LocalLLMClient:
        """Create a local LLM client.
        
        Args:
            model_id (str): HuggingFace model identifier.
            local_path (str): Local path for model storage.
            device (str): Device to use ('auto', 'cuda', 'cpu').
                Defaults to 'auto'.
            **kwargs: Additional parameters passed to LocalLLMClient.
        
        Returns:
            LocalLLMClient: Configured LocalLLMClient instance.
        """
        return LocalLLMClient(
            model_id=model_id,
            local_path=local_path,
            device=device,
            **kwargs
        )
    
    @staticmethod
    def create_paratera(
        api_key: str,
        model_name: str = "Qwen3-Next-80B-A3B-Thinking",
        **kwargs
    ) -> OnlineLLMClient:
        """Create a client for Paratera API (commonly used in the project).
        
        Args:
            api_key (str): Paratera API key.
            model_name (str): Model name to use.
                Defaults to 'Qwen3-Next-80B-A3B-Thinking'.
            **kwargs: Additional parameters.
        
        Returns:
            OnlineLLMClient: Configured OnlineLLMClient for Paratera.
        """
        return OnlineLLMClient(
            api_key=api_key,
            api_type="paratera",
            model_name=model_name,
            **kwargs
        )


def create_llm_client(
    client_type: str,
    **kwargs
) -> BaseLLMClient:
    """Convenience function to create an LLM client.
    
    Args:
        client_type (str): Type of client ('online' or 'local').
        **kwargs: Configuration parameters for the client.
    
    Returns:
        BaseLLMClient: Configured LLM client instance.
    
    Raises:
        ValueError: If client_type is not recognized.
    
    Example:
        >>> # Online client
        >>> client = create_llm_client(
        ...     "online",
        ...     api_key="sk-xxx",
        ...     api_type="openai"
        ... )
        
        >>> # Local client
        >>> client = create_llm_client(
        ...     "local",
        ...     model_id="Qwen/Qwen2.5-7B-Instruct",
        ...     local_path="./models"
        ... )
    """
    if client_type.lower() == "online":
        return LLMClientFactory.create_online(**kwargs)
    elif client_type.lower() == "local":
        return LLMClientFactory.create_local(**kwargs)
    else:
        raise ValueError(
            f"Unknown client type: {client_type}. "
            f"Expected 'online' or 'local'."
        )
