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
