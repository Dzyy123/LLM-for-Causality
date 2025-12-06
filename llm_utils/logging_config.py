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

# Default logging formats
# Concise format with date and time: MM-DD HH:MM:SS
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
# Detailed format with full timestamp
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
# Simple format without timestamp (for cases where timestamps aren't needed)
SIMPLE_FORMAT = "[%(levelname)s] %(name)s - %(message)s"

# Default date format: concise month-day hour:minute:second
DEFAULT_DATE_FORMAT = "%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    use_detailed_format: bool = False,
    no_timestamps: bool = False,
    stream: Optional[object] = None
) -> logging.Logger:
    """Set up logging configuration for the llm_utils package.
    
    This function configures logging for the entire llm_utils package.
    It should be called once at the start of your application.
    
    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 
            'CRITICAL'). Defaults to 'INFO'.
        format_string (Optional[str]): Custom format string for log messages.
            If None, uses default format with timestamps.
        date_format (Optional[str]): Custom date format string. Defaults to
            "%m-%d %H:%M:%S" (e.g., "12-06 14:30:45").
        use_detailed_format (bool): If True and format_string is None, uses
            a detailed format with function names and line numbers. 
            Defaults to False.
        no_timestamps (bool): If True and format_string is None, uses
            a simple format without timestamps. Defaults to False.
        stream (Optional[object]): Stream to output logs to. Defaults to
            sys.stdout.
    
    Returns:
        logging.Logger: The configured logger instance.
    
    Example:
        >>> # Basic setup with timestamps (default)
        >>> logger = setup_logging()
        >>> # Output: 12-06 14:30:45 [INFO] llm_utils - Message
        
        >>> # Debug level with detailed format
        >>> logger = setup_logging(level="DEBUG", use_detailed_format=True)
        >>> # Output: 12-06 14:30:45 [DEBUG] llm_utils.module:func:42 - Message
        
        >>> # Simple format without timestamps
        >>> logger = setup_logging(no_timestamps=True)
        >>> # Output: [INFO] llm_utils - Message
        
        >>> # Custom date format with full year
        >>> logger = setup_logging(date_format="%Y-%m-%d %H:%M:%S")
        >>> # Output: 2025-12-06 14:30:45 [INFO] llm_utils - Message
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
        if no_timestamps:
            format_string = SIMPLE_FORMAT
        elif use_detailed_format:
            format_string = DETAILED_FORMAT
        else:
            format_string = DEFAULT_FORMAT
    
    # Determine date format
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
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
        12-06 14:30:45 [INFO] llm_utils.online_client - Client initialized
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


# Auto-setup with default configuration when package is imported
# Users can call setup_logging() again to customize settings
_default_logger = setup_logging()
