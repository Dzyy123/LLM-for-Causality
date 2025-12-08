"""
Data types and structures for LLM interactions.

This module contains dataclasses and enums used throughout the LLM utilities
package for standardized data representation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ResponseFormat(Enum):
    """Enumeration of supported response formats."""
    
    TEXT = "text"
    JSON = "json"
    PROBABILITY = "probability"


@dataclass
class LLMResponse:
    """Standardized response object from LLM interactions.
    
    Attributes:
        content (str): The text content of the response.
        raw_response (Any): The raw response object from the API/model.
        token_probabilities (Optional[Dict[str, float]]): Optional token-level
            probabilities.
        metadata (Dict[str, Any]): Additional metadata about the response.
    """
    
    content: str
    raw_response: Any = None
    token_probabilities: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


    def crop_thinking(self) -> "LLMResponse":
        """Crop the thinking part (before </think> tag) from the response.
        
        This method creates a new LLMResponse with:
        - Content after the </think> tag
        - Token probabilities after the </think> tag
        - Updated metadata with cropped information
        
        The original response remains unchanged.
        
        Returns:
            LLMResponse: A new response object with thinking part removed.
                If </think> tag is not found, returns a copy of the original.
        
        Example:
            >>> response = client.chat("What is 2+2?", return_token_probs=True)
            >>> print(response.content)  # "<think>...</think>\n4"
            >>> cropped = response.crop_thinking()
            >>> print(cropped.content)  # "4"
            >>> probs = client.get_token_probabilities(cropped)
        """
        from copy import deepcopy
        
        # Find </think> in content
        content = self.content
        think_tag = "</think>"
        think_idx = content.lower().find(think_tag.lower())
        
        if think_idx == -1:
            # No </think> tag found, return a deep copy
            return LLMResponse(
                content=content,
                raw_response=self.raw_response,
                token_probabilities=self.token_probabilities,
                metadata=deepcopy(self.metadata)
            )
        
        # Crop content after </think>
        cropped_content = content[think_idx + len(think_tag):].lstrip()
        
        # Crop token probabilities if present
        cropped_metadata = deepcopy(self.metadata)
        if "token_probabilities" in self.metadata:
            token_probs = self.metadata["token_probabilities"]
            
            # Find the index where </think> appears in tokens
            think_end_idx = None
            for i, item in enumerate(token_probs):
                token = item["token"]
                # Check for various forms of the closing think tag
                if "</think>" in token.lower() or token.strip().lower() == "</think>":
                    think_end_idx = i + 1  # Start from the next token
                    break
            
            # Crop token probabilities
            if think_end_idx is not None:
                cropped_token_probs = token_probs[think_end_idx:]
                cropped_metadata["token_probabilities"] = cropped_token_probs
                cropped_metadata["thinking_tokens_removed"] = think_end_idx
                
                # Also crop token_distributions if present
                if "token_distributions" in self.metadata:
                    token_dists = self.metadata["token_distributions"]
                    cropped_token_dists = token_dists[think_end_idx:]
                    cropped_metadata["token_distributions"] = cropped_token_dists
                
                # Update output_tokens count if present
                if "output_tokens" in cropped_metadata:
                    original_count = cropped_metadata["output_tokens"]
                    cropped_metadata["output_tokens"] = len(cropped_token_probs)
                    cropped_metadata["original_output_tokens"] = original_count
        
        return LLMResponse(
            content=cropped_content,
            raw_response=self.raw_response,
            token_probabilities=self.token_probabilities,
            metadata=cropped_metadata
        )


@dataclass
class ChatMessage:
    """Represents a single chat message.
    
    Attributes:
        role (str): The role of the message sender (system, user, assistant).
        content (str): The content of the message.
    """
    
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format.
        
        Returns:
            Dict[str, str]: Dictionary representation of the message.
        """
        return {"role": self.role, "content": self.content}
