"""
Conversation management for LLM interactions.

This module provides a Conversation class for managing multi-turn dialogues
with automatic history tracking and context management.
"""

from typing import List, Optional, TYPE_CHECKING
from copy import deepcopy

from llm_utils.data_types import ChatMessage, LLMResponse
from llm_utils.logging_config import get_logger

if TYPE_CHECKING:
    from llm_utils.base_client import BaseLLMClient


logger = get_logger("conversation")


class Conversation:
    """Manages a multi-turn conversation with automatic history tracking.
    
    This class maintains conversation history and provides convenient methods
    for sending messages while automatically managing the context.
    
    Attributes:
        client (BaseLLMClient): The LLM client used for generating responses.
        system_prompt (Optional[str]): The system prompt for the conversation.
        messages (List[ChatMessage]): The conversation history.
        max_history (Optional[int]): Maximum number of messages to keep in history.
            If set, older messages are removed when limit is reached (system message
            is always kept).
    
    Example:
        >>> client = LocalLLMClient(model_id="Qwen/Qwen2.5-7B", ...)
        >>> conv = client.start_conversation(system_prompt="You are a helpful assistant.")
        >>> 
        >>> response = conv.send("What is 2+2?")
        >>> print(response.content)  # "4"
        >>> 
        >>> response = conv.send("What is that number squared?")
        >>> print(response.content)  # "16" (uses previous context)
        >>> 
        >>> # Inspect history
        >>> for msg in conv.get_history():
        ...     print(f"{msg.role}: {msg.content}")
    """
    
    def __init__(
        self,
        client: "BaseLLMClient",
        system_prompt: Optional[str] = None,
        max_history: Optional[int] = None
    ) -> None:
        """Initialize a new conversation.
        
        Args:
            client (BaseLLMClient): The LLM client to use for responses.
            system_prompt (Optional[str]): Optional system prompt to set context.
            max_history (Optional[int]): Maximum number of messages to keep.
                If None, keeps all messages. If set, older messages (excluding
                system message) are removed when limit is reached.
        """
        self.client = client
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        
        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))
            logger.info("Conversation started with system prompt")
        else:
            logger.info("Conversation started without system prompt")
    
    def send(
        self,
        user_message: str,
        **kwargs
    ) -> LLMResponse:
        """Send a user message and get a response.
        
        This method automatically:
        1. Adds the user message to history
        2. Sends the conversation history to the LLM
        3. Adds the assistant's response to history
        4. Manages history length if max_history is set
        
        Args:
            user_message (str): The user's message.
            **kwargs: Additional parameters to pass to chat_with_history().
        
        Returns:
            LLMResponse: The assistant's response.
        
        Example:
            >>> response = conv.send("Hello!")
            >>> print(response.content)
        """
        # Add user message
        self.messages.append(ChatMessage(role="user", content=user_message))
        logger.info(f"User message added to conversation (total messages: {len(self.messages)})")
        
        # Get response from LLM
        response = self.client.chat_with_history(self.messages, **kwargs)
        
        # Add assistant response
        self.messages.append(ChatMessage(role="assistant", content=response.content))
        logger.info(f"Assistant response added to conversation (total messages: {len(self.messages)})")
        
        # Manage history length if needed
        self._trim_history()
        
        return response
    
    def _trim_history(self) -> None:
        """Trim conversation history if it exceeds max_history."""
        if self.max_history is None:
            return
        
        # Keep system message + last max_history messages
        has_system = self.messages and self.messages[0].role == "system"
        
        if has_system:
            # Keep system message + last (max_history - 1) messages
            if len(self.messages) > self.max_history:
                system_msg = self.messages[0]
                recent_messages = self.messages[-(self.max_history - 1):]
                removed_count = len(self.messages) - len(recent_messages) - 1
                self.messages = [system_msg] + recent_messages
                logger.info(f"Trimmed {removed_count} old messages from history")
        else:
            # Keep last max_history messages
            if len(self.messages) > self.max_history:
                removed_count = len(self.messages) - self.max_history
                self.messages = self.messages[-self.max_history:]
                logger.info(f"Trimmed {removed_count} old messages from history")
    
    def get_history(self) -> List[ChatMessage]:
        """Get a copy of the conversation history.
        
        Returns:
            List[ChatMessage]: A copy of all messages in the conversation.
        
        Example:
            >>> history = conv.get_history()
            >>> for msg in history:
            ...     print(f"{msg.role}: {msg.content}")
        """
        return deepcopy(self.messages)
    
    def clear_history(self, keep_system: bool = True) -> None:
        """Clear the conversation history.
        
        Args:
            keep_system (bool): If True, keeps the system message (if present).
                Defaults to True.
        
        Example:
            >>> conv.clear_history()  # Clears all except system message
            >>> conv.clear_history(keep_system=False)  # Clears everything
        """
        if keep_system and self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
            self.messages = [system_msg]
            logger.info("Conversation history cleared (system message kept)")
        else:
            self.messages = []
            logger.info("Conversation history cleared completely")
    
    def add_message(self, role: str, content: str) -> None:
        """Manually add a message to the conversation history.
        
        This is useful for:
        - Injecting context from external sources
        - Manually correcting conversation history
        - Building hypothetical conversation states
        
        Args:
            role (str): The role of the message sender ("system", "user", "assistant").
            content (str): The content of the message.
        
        Example:
            >>> # Inject external context
            >>> conv.add_message("user", "What did we discuss earlier?")
            >>> conv.add_message("assistant", "We discussed causality.")
        """
        self.messages.append(ChatMessage(role=role, content=content))
        logger.info(f"Manual message added ({role}): {len(content)} chars")
        self._trim_history()
    
    def remove_last_exchange(self) -> bool:
        """Remove the last user-assistant exchange from history.
        
        This is useful for:
        - Undoing mistakes
        - Rolling back to previous conversation state
        - Retrying with different parameters
        
        Returns:
            bool: True if an exchange was removed, False if not enough messages.
        
        Example:
            >>> conv.send("Wrong question")
            >>> conv.remove_last_exchange()  # Undo
            >>> conv.send("Correct question")
        """
        # Find the last assistant message
        assistant_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                assistant_idx = i
                break
        
        if assistant_idx is None:
            logger.warning("No assistant message found to remove")
            return False
        
        # Find the corresponding user message (should be right before)
        user_idx = None
        for i in range(assistant_idx - 1, -1, -1):
            if self.messages[i].role == "user":
                user_idx = i
                break
        
        if user_idx is None:
            logger.warning("No user message found before assistant message")
            return False
        
        # Remove both messages
        self.messages = self.messages[:user_idx] + self.messages[assistant_idx + 1:]
        logger.info("Removed last user-assistant exchange")
        return True
    
    def get_message_count(self) -> int:
        """Get the total number of messages in the conversation.
        
        Returns:
            int: Total number of messages (including system message if present).
        """
        return len(self.messages)
    
    def get_token_estimate(self) -> int:
        """Estimate the total number of tokens in the conversation.
        
        This is a rough estimate based on character count. For accurate
        token counting, use the tokenizer from your specific model.
        
        Returns:
            int: Estimated token count (roughly characters / 4).
        """
        total_chars = sum(len(msg.content) for msg in self.messages)
        # Rough estimate: 1 token ≈ 4 characters for English text
        return total_chars // 4
    
    def __len__(self) -> int:
        """Get the number of messages in the conversation.
        
        Returns:
            int: Number of messages.
        """
        return len(self.messages)
    
    def __str__(self) -> str:
        """Get a string representation of the conversation.
        
        Returns:
            str: Formatted conversation history.
        """
        lines = [f"Conversation ({len(self.messages)} messages):"]
        for msg in self.messages:
            lines.append(f"  {msg.role}: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
        return "\n".join(lines)
