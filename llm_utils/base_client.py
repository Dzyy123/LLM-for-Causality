"""
Base client class for LLM interactions.

This module contains the abstract base class that defines the common interface
for all LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import math
import re

from llm_utils.data_types import LLMResponse, ChatMessage

if TYPE_CHECKING:
    from llm_utils.conversation import Conversation


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.
    
    This class defines the common interface that all LLM clients must implement,
    ensuring consistent behavior across different model backends.
    
    Attributes:
        model_name (str): Name or identifier of the model.
        temperature (float): Sampling temperature for response generation.
        max_tokens (int): Maximum number of tokens in the response.
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> None:
        """Initialize the base LLM client.
        
        Args:
            model_name (str): Name or identifier of the model.
            temperature (float): Sampling temperature for response generation.
                Defaults to 0.7.
            max_tokens (int): Maximum number of tokens in the response.
                Defaults to 200.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._is_initialized = False
    
    @abstractmethod
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
            **kwargs: Additional model-specific parameters.
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        pass
    
    @abstractmethod
    def chat_with_history(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """Send a conversation with history and get a response.
        
        Args:
            messages (List[ChatMessage]): List of chat messages representing
                the conversation.
            **kwargs: Additional model-specific parameters.
        
        Returns:
            LLMResponse: Standardized LLM response object.
        """
        pass
    
    @abstractmethod
    def get_token_probabilities(
        self,
        prompt: str,
        target_tokens: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get token probabilities for the next token prediction.
        
        Args:
            prompt (str): The input prompt.
            target_tokens (Optional[List[str]]): Optional list of tokens to get
                probabilities for.
        
        Returns:
            Dict[str, float]: Dictionary mapping tokens to their probabilities.
        """
        pass
    
    def start_conversation(
        self,
        system_prompt: Optional[str] = None,
        max_history: Optional[int] = None
    ) -> "Conversation":
        """Start a new conversation with automatic history management.
        
        This method creates a Conversation object that automatically tracks
        message history and provides convenient methods for multi-turn dialogues.
        
        Args:
            system_prompt (Optional[str]): Optional system prompt to set context.
            max_history (Optional[int]): Maximum number of messages to keep in history.
                If None, keeps all messages. If set, older messages (excluding
                system message) are removed when limit is reached.
        
        Returns:
            Conversation: A new conversation manager instance.
        
        Example:
            >>> client = LocalLLMClient(...)
            >>> conv = client.start_conversation(system_prompt="You are helpful.")
            >>> response = conv.send("What is 2+2?")
            >>> response = conv.send("What is that squared?")  # Uses context
            >>> print(conv.get_history())  # Inspect full conversation
        """
        from llm_utils.conversation import Conversation
        return Conversation(self, system_prompt=system_prompt, max_history=max_history)
    
    def judge_binary(
        self,
        prompt: str,
        method: str = "frequency",
        n_samples: int = 10
    ) -> Dict[str, Any]:
        """Make a binary judgment (yes/no) with probability estimation.
        
        Args:
            prompt (str): The question or prompt to judge.
            method (str): Method for probability estimation. One of 'frequency',
                'probability', or 'logit'. Defaults to 'frequency'.
            n_samples (int): Number of samples for frequency-based estimation.
                Defaults to 10.
        
        Returns:
            Dict[str, Any]: Dictionary with 'label' (0 or 1) and 'prob'
                (confidence).
        
        Raises:
            ValueError: If method is not one of the supported methods.
        """
        if method == "frequency":
            return self._judge_by_frequency(prompt, n_samples)
        elif method == "probability":
            return self._judge_by_probability(prompt)
        elif method == "logit":
            return self._judge_by_logit(prompt)
        else:
            raise ValueError(
                f"Method must be one of 'frequency', 'probability', or 'logit', "
                f"got '{method}'"
            )
    
    def _judge_by_frequency(
        self,
        prompt: str,
        n_samples: int
    ) -> Dict[str, Any]:
        """Judge by sampling multiple responses and counting votes.
        
        Args:
            prompt (str): The question to judge.
            n_samples (int): Number of samples to collect.
        
        Returns:
            Dict[str, Any]: Dictionary with label and probability.
        """
        votes = []
        for _ in range(n_samples):
            response = self.chat(prompt, temperature=0.7, max_tokens=50)
            text = response.content.strip().lower()
            votes.append(1 if text.startswith("yes") else 0)
        
        p_yes = sum(votes) / len(votes)
        p_no = 1 - p_yes
        
        if p_yes >= p_no:
            return {"label": 1, "prob": p_yes}
        else:
            return {"label": 0, "prob": p_no}
    
    def _judge_by_probability(self, prompt: str) -> Dict[str, Any]:
        """Judge by parsing probability values from response.
        
        Args:
            prompt (str): The question to judge.
        
        Returns:
            Dict[str, Any]: Dictionary with label and probability.
        """
        response = self.chat(prompt, temperature=0.7, max_tokens=200)
        text = response.content.strip()
        
        p_yes, p_no = self._parse_probabilities(text)
        
        if p_yes >= p_no:
            return {"label": 1, "prob": p_yes}
        else:
            return {"label": 0, "prob": p_no}
    
    def _judge_by_logit(self, prompt: str) -> Dict[str, Any]:
        """Judge by parsing logit values from response and converting to probabilities.
        
        Args:
            prompt (str): The question to judge.
        
        Returns:
            Dict[str, Any]: Dictionary with label and probability.
        """
        response = self.chat(prompt, temperature=0.7, max_tokens=200)
        text = response.content.strip()
        
        logit_yes, logit_no = self._parse_logits(text)
        p_yes = self._sigmoid(logit_yes)
        p_no = self._sigmoid(logit_no)
        
        if p_yes >= p_no:
            return {"label": 1, "prob": p_yes}
        else:
            return {"label": 0, "prob": p_no}
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Compute sigmoid function.
        
        Args:
            x (float): Input value.
        
        Returns:
            float: Sigmoid of x.
        """
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def _parse_probabilities(text: str) -> Tuple[float, float]:
        """Parse probability values from LLM output text.
        
        Args:
            text (str): Text containing probability values.
        
        Returns:
            Tuple[float, float]: Tuple of (p_yes, p_no) probabilities.
        """
        matches = re.findall(r"([0-9]*\.?[0-9]+)", text)
        if len(matches) >= 2:
            p1, p2 = float(matches[0]), float(matches[1])
            total = p1 + p2
            if total == 0:
                return 0.5, 0.5
            return p1 / total, p2 / total
        return 0.5, 0.5
    
    @staticmethod
    def _parse_logits(text: str) -> Tuple[float, float]:
        """Parse logit values from LLM output text.
        
        Args:
            text (str): Text containing logit values.
        
        Returns:
            Tuple[float, float]: Tuple of (logit_yes, logit_no) values.
        """
        matches = re.findall(r"([-]?[0-9]*\.?[0-9]+)", text)
        if len(matches) >= 2:
            l1, l2 = float(matches[0]), float(matches[1])
            if not math.isclose(l1 + l2, 0):
                avg = (l1 - l2) / 2
                l1, l2 = avg, -avg
            return l1, l2
        return 0.0, 0.0
    
    def __repr__(self) -> str:
        """Return string representation of the client.
        
        Returns:
            str: String representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
