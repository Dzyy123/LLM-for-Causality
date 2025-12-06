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
            logger.info("Model found and validated at local path: %s", self.local_path)
            return self.local_path
        
        logger.info("Model not found locally or invalid, downloading: %s", self.model_id)
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
                return_token_probs (bool): Includes token probabilities in the response metadata. Defaults to True.
        
        Returns:
            LLMResponse: Standardized LLM response object with optional token
                probabilities in metadata['token_probabilities'].
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return_token_probs = kwargs.get("return_token_probs", True)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]
        
        if return_token_probs:
            # Generate with output_scores to get token probabilities
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            generated_tokens = outputs.sequences[0][input_length:]
            content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract token probabilities
            token_probs = []
            for i, score in enumerate(outputs.scores):
                probs = F.softmax(score[0], dim=-1)
                token_id = generated_tokens[i].item()
                token_str = self._tokenizer.decode([token_id])
                token_prob = probs[token_id].item()
                token_probs.append({
                    "token": token_str,
                    "token_id": token_id,
                    "probability": token_prob
                })
            
            metadata = {
                "model": self.model_id,
                "input_tokens": input_length,
                "output_tokens": len(generated_tokens),
                "token_probabilities": token_probs
            }
        else:
            # Standard generation without probabilities
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][input_length:]
            content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            metadata = {
                "model": self.model_id,
                "input_tokens": input_length,
                "output_tokens": len(generated_tokens)
            }
        
        return LLMResponse(
            content=content.strip(),
            raw_response=outputs,
            metadata=metadata
        )
    
    def get_token_probabilities(
        self,
        response: LLMResponse,
        aggregate: str = "first",
        skip_thinking: bool = False
    ) -> Dict[str, float]:
        """Extract token probabilities from a generated response.
        
        This method extracts the probability of each token that was generated
        in the response. The response must have been generated with the
        return_token_probs=True parameter.
        
        Args:
            response (LLMResponse): The response object containing token probabilities.
                Must have 'token_probabilities' in metadata.
            aggregate (str): How to aggregate probabilities for repeated tokens.
                - 'first': Use the first occurrence probability (default)
                - 'max': Use the maximum probability across occurrences
                - 'mean': Use the average probability across occurrences
                - 'all': Return all occurrences as a list
            skip_thinking (bool): If True, only consider tokens after the '</think>' 
                closing tag for Chain-of-Thought (CoT) models. This filters out the 
                reasoning process tokens and focuses on the final answer tokens.
                Uses response.crop_thinking() internally. Defaults to False.
        
        Returns:
            Dict[str, float]: Dictionary mapping tokens to their probabilities.
                If aggregate='all', values are lists of floats.
        
        Raises:
            ValueError: If response doesn't contain token probabilities or
                invalid aggregate method is specified.
        
        Example:
            >>> # Standard usage
            >>> response = client.chat("Hello", return_token_probs=True)
            >>> probs = client.get_token_probabilities(response)
            >>> print(probs)  # {'Hello': 0.95, '!': 0.87, ...}
            
            >>> # For thinking models (CoT) - Method 1: use skip_thinking
            >>> response = client.chat("What is 2+2?", return_token_probs=True)
            >>> probs = client.get_token_probabilities(response, skip_thinking=True)
            >>> print(probs)  # Only tokens after </think>: {'4': 0.92, ...}
            
            >>> # For thinking models (CoT) - Method 2: crop first
            >>> cropped = response.crop_thinking()
            >>> probs = client.get_token_probabilities(cropped)
        """
        # Use crop_thinking() if skip_thinking is True
        if skip_thinking:
            response = response.crop_thinking()
            if "thinking_tokens_removed" in response.metadata:
                removed_count = response.metadata["thinking_tokens_removed"]
                logger.info(f"Cropped {removed_count} thinking tokens from response")
        
        if "token_probabilities" not in response.metadata:
            raise ValueError(
                "Response does not contain token probabilities. "
                "Generate response with return_token_probs=True parameter."
            )
        
        token_probs = response.metadata["token_probabilities"]
        
        if aggregate == "all":
            # Return all occurrences
            result = {}
            for item in token_probs:
                token = item["token"]
                prob = item["probability"]
                if token not in result:
                    result[token] = []
                result[token].append(prob)
            return result
        
        elif aggregate == "first":
            # Use first occurrence
            result = {}
            for item in token_probs:
                token = item["token"]
                if token not in result:
                    result[token] = item["probability"]
            return result
        
        elif aggregate == "max":
            # Use maximum probability
            result = {}
            for item in token_probs:
                token = item["token"]
                prob = item["probability"]
                if token not in result or prob > result[token]:
                    result[token] = prob
            return result
        
        elif aggregate == "mean":
            # Use average probability
            token_lists = {}
            for item in token_probs:
                token = item["token"]
                prob = item["probability"]
                if token not in token_lists:
                    token_lists[token] = []
                token_lists[token].append(prob)
            
            result = {
                token: sum(probs) / len(probs)
                for token, probs in token_lists.items()
            }
            return result
        
        else:
            raise ValueError(
                f"Invalid aggregate method: {aggregate}. "
                "Must be 'first', 'max', 'mean', or 'all'."
            )
    
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
