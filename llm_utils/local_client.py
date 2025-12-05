"""
Local LLM client for HuggingFace models.

This module provides a client for loading and interacting with local
HuggingFace models, including automatic downloading and validation.
"""

from typing import Any, Dict, List, Optional
import os

import torch
import torch.nn.functional as F

from llm_utils.logging_config import get_logger
from llm_utils.base_client import BaseLLMClient
from llm_utils.data_types import LLMResponse, ChatMessage


# Module-level logger
logger = get_logger("local_client")


class LocalLLMClient(BaseLLMClient):
    """Client for local HuggingFace models.
    
    This class handles loading, validation, and inference with local LLM models.
    It automatically downloads models from HuggingFace Hub if not present locally.
    
    Attributes:
        model_id (str): HuggingFace model identifier.
        local_path (str): Local directory path for model storage.
        device (str): Device to run the model on ('auto', 'cuda', 'cpu').
        dtype (torch.dtype): Data type for model weights.
        mirror_url (Optional[str]): Mirror URL for HuggingFace Hub downloads.
            Use 'https://hf-mirror.com' for Chinese users.
    
    Example:
        >>> # Default download from huggingface.co
        >>> client = LocalLLMClient(
        ...     model_id="Qwen/Qwen2.5-7B-Instruct",
        ...     local_path="./models/qwen",
        ...     device="auto"
        ... )
        >>> response = client.chat("Explain causality in simple terms.")
        
        >>> # Download from mirror site (for Chinese users)
        >>> client = LocalLLMClient(
        ...     model_id="Qwen/Qwen2.5-7B-Instruct",
        ...     local_path="./models/qwen",
        ...     mirror_url="https://hf-mirror.com"
        ... )
    """
    
    # Required files for model validation
    REQUIRED_CONFIG_FILES = ["config.json"]
    MODEL_FILE_PATTERNS = ["model", "pytorch", "safetensors"]
    
    # Default HuggingFace Hub endpoint
    DEFAULT_HF_ENDPOINT = "https://huggingface.co"
    
    def __init__(
        self,
        model_id: str,
        local_path: str,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        load_on_init: bool = True,
        mirror_url: Optional[str] = None
    ) -> None:
        """Initialize the local LLM client.
        
        Args:
            model_id (str): HuggingFace model identifier.
            local_path (str): Local directory path for model storage.
            device (str): Device to run the model on ('auto', 'cuda', 'cpu').
                Defaults to 'auto'.
            dtype (Optional[torch.dtype]): Data type for model weights. If None,
                uses float16 on CUDA and float32 on CPU.
            temperature (float): Sampling temperature for generation.
                Defaults to 0.7.
            max_tokens (int): Maximum number of tokens in response.
                Defaults to 200.
            load_on_init (bool): Whether to load the model during initialization.
                Defaults to True.
            mirror_url (Optional[str]): Mirror URL for HuggingFace Hub downloads.
                Defaults to None (uses official huggingface.co).
                Common mirrors:
                - 'https://hf-mirror.com' (China mirror)
        """
        super().__init__(model_id, temperature, max_tokens)
        
        self.model_id = model_id
        self.local_path = local_path
        self.device = device
        self.dtype = dtype or (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.mirror_url = mirror_url
        
        self._model = None
        self._tokenizer = None
        
        if load_on_init:
            self._ensure_model_ready()
            self._load_model()
    
    def _ensure_model_ready(self) -> str:
        """Ensure the model is downloaded and validated.
        
        Returns:
            str: Path to the validated model directory.
        
        Raises:
            RuntimeError: If model download or validation fails.
        """
        if self._validate_local_model():
            logger.info("Model found at local path: %s", self.local_path)
            return self.local_path
        
        logger.info("Model not found locally, downloading: %s", self.model_id)
        return self._download_model()
    
    def _validate_local_model(self) -> bool:
        """Validate that the local model directory contains required files.
        
        Returns:
            bool: True if model is valid, False otherwise.
        """
        if not os.path.exists(self.local_path):
            return False
        
        files = os.listdir(self.local_path)
        
        # Check for config files
        has_config = any(
            config_file in files 
            for config_file in self.REQUIRED_CONFIG_FILES
        )
        
        # Check for model weight files
        has_model_weights = any(
            any(pattern in f for pattern in self.MODEL_FILE_PATTERNS)
            for f in files
        )
        
        is_valid = has_config and has_model_weights
        
        if is_valid:
            logger.info("Model validation successful: %s", self.local_path)
        else:
            if not has_config:
                logger.warning("Missing config files in %s", self.local_path)
            if not has_model_weights:
                logger.warning("Missing model weight files in %s", self.local_path)
        
        return is_valid
    
    def _download_model(self) -> str:
        """Download model from HuggingFace Hub.
        
        Uses the mirror_url if specified, otherwise uses the default
        HuggingFace Hub endpoint. The mirror is set via the HF_ENDPOINT
        environment variable which is respected by huggingface_hub.
        
        Returns:
            str: Path to the downloaded model directory.
        
        Raises:
            RuntimeError: If download fails.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub package is required for model download. "
                "Install with: pip install huggingface_hub"
            ) from e
        
        # Set mirror URL via environment variable if specified
        original_endpoint = os.environ.get("HF_ENDPOINT")
        if self.mirror_url:
            os.environ["HF_ENDPOINT"] = self.mirror_url
            logger.info("Using mirror: %s", self.mirror_url)
        
        download_source = self.mirror_url or self.DEFAULT_HF_ENDPOINT
        logger.info("Downloading model from %s: %s", download_source, self.model_id)
        logger.info("Target directory: %s", self.local_path)
        
        os.makedirs(self.local_path, exist_ok=True)
        
        try:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.local_path,
                local_dir_use_symlinks=False,  # Windows compatibility
                resume_download=True
            )
            logger.info("Model download completed: %s", self.local_path)
            return self.local_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model '{self.model_id}': {e}"
            ) from e
        finally:
            # Restore original HF_ENDPOINT
            if self.mirror_url:
                if original_endpoint is not None:
                    os.environ["HF_ENDPOINT"] = original_endpoint
                else:
                    os.environ.pop("HF_ENDPOINT", None)
    
    def _load_model(self) -> None:
        """Load the model and tokenizer into memory.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers package is required for local models. "
                "Install with: pip install transformers"
            ) from e
        
        logger.info("Loading tokenizer from: %s", self.local_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.local_path)
        
        logger.info("Loading model from: %s", self.local_path)
        logger.info("Device: %s, Dtype: %s", self.device, self.dtype)
        
        device_map = self.device if self.device != "auto" else "auto"
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.local_path,
            dtype=self.dtype,
            device_map=device_map
        )
        
        self._is_initialized = True
        logger.info("Local LLM client initialized: %s", self.model_id)
    
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
            **kwargs: Additional generation parameters.
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call _load_model() first.")
        
        # Build conversation
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template if available (check both method and template attribute)
        has_chat_template = (
            hasattr(self._tokenizer, "apply_chat_template") and 
            hasattr(self._tokenizer, "chat_template") and
            self._tokenizer.chat_template is not None
        )
        
        if has_chat_template:
            full_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template (base language models)
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
        
        return self._generate(full_prompt, **kwargs)
    
    def chat_with_history(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """Send a conversation with history and get a response.
        
        Args:
            messages (List[ChatMessage]): List of chat messages representing
                the conversation.
            **kwargs: Additional generation parameters.
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call _load_model() first.")
        
        formatted_messages = [msg.to_dict() for msg in messages]
        
        # Check if chat template is available
        has_chat_template = (
            hasattr(self._tokenizer, "apply_chat_template") and 
            hasattr(self._tokenizer, "chat_template") and
            self._tokenizer.chat_template is not None
        )
        
        if has_chat_template:
            full_prompt = self._tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            full_prompt = "\n".join(
                f"{msg.role}: {msg.content}" for msg in messages
            )
        
        return self._generate(full_prompt, **kwargs)
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the model.
        
        Args:
            prompt (str): The formatted prompt string.
            **kwargs: Additional generation parameters.
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][input_length:]
        content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return LLMResponse(
            content=content.strip(),
            raw_response=outputs,
            metadata={
                "model": self.model_id,
                "input_tokens": input_length,
                "output_tokens": len(generated_tokens)
            }
        )
    
    def get_token_probabilities(
        self,
        prompt: str,
        target_tokens: Optional[List[str]] = None,
        top_k: int = 20
    ) -> Dict[str, float]:
        """Get token probabilities for the next token prediction.
        
        Args:
            prompt (str): The input prompt.
            target_tokens (Optional[List[str]]): Optional list of specific tokens
                to get probabilities for. If None, returns top_k most likely tokens.
            top_k (int): Number of top tokens to return if target_tokens is None.
                Defaults to 20.
        
        Returns:
            Dict[str, float]: Dictionary mapping tokens to their probabilities.
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call _load_model() first.")
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        result = {}
        
        if target_tokens:
            # Get probabilities for specific tokens
            for token in target_tokens:
                token_ids = self._tokenizer.encode(token, add_special_tokens=False)
                total_prob = sum(
                    next_token_probs[tid].item()
                    for tid in token_ids
                    if tid < len(next_token_probs)
                )
                result[token] = total_prob
        else:
            # Get top_k tokens
            probs, indices = torch.topk(next_token_probs, top_k)
            for prob, idx in zip(probs, indices):
                token_str = self._tokenizer.decode([idx.item()])
                result[token_str.strip()] = prob.item()
        
        return result
    
    def unload_model(self) -> None:
        """Unload the model from memory to free up resources."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_initialized = False
        logger.info("Model unloaded: %s", self.model_id)
