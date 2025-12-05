"""
Logging configuration for the llm_utils package.

This module provides centralized logging setup for all modules in the package.
Import and call setup_logging() once at application startup to configure logging.

Example:
    >>> from llm_utils.logging_config import setup_logging
    >>> setup_logging(level="DEBUG")  # Enable debug output
    
    >>> # Or use default INFO level
    >>> setup_logging()
"""

import logging
import sys
from typing import Optional


# Package-level logger name
LOGGER_NAME = "llm_utils"

# Default logging format
DEFAULT_FORMAT = "[%(levelname)s] %(name)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    use_detailed_format: bool = False,
    stream: Optional[object] = None
) -> logging.Logger:
    """Set up logging configuration for the llm_utils package.
    
    This function configures logging for the entire llm_utils package.
    It should be called once at the start of your application.
    
    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 
            'CRITICAL'). Defaults to 'INFO'.
        format_string (Optional[str]): Custom format string for log messages.
            If None, uses default format.
        use_detailed_format (bool): If True and format_string is None, uses
            a more detailed format with timestamps. Defaults to False.
        stream (Optional[object]): Stream to output logs to. Defaults to
            sys.stdout.
    
    Returns:
        logging.Logger: The configured logger instance.
    
    Example:
        >>> # Basic setup
        >>> logger = setup_logging()
        
        >>> # Debug level with timestamps
        >>> logger = setup_logging(level="DEBUG", use_detailed_format=True)
        
        >>> # Custom format
        >>> logger = setup_logging(
        ...     format_string="%(asctime)s - %(levelname)s - %(message)s"
        ... )
    """
    # Get or create the package logger
    logger = logging.getLogger(LOGGER_NAME)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set the logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Determine format string
    if format_string is None:
        format_string = DETAILED_FORMAT if use_detailed_format else DEFAULT_FORMAT
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create stream handler
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name (Optional[str]): Module name to append to the package logger name.
            If None, returns the package-level logger.
    
    Returns:
        logging.Logger: Logger instance for the specified module.
    
    Example:
        >>> logger = get_logger("online_client")
        >>> logger.info("Client initialized")
        [INFO] llm_utils.online_client - Client initialized
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


# Auto-setup with default configuration when package is imported
# Users can call setup_logging() again to customize settings
_default_logger = setup_logging()
