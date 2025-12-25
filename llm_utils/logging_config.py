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
from rich.logging import RichHandler
from rich.console import Console


# Package-level logger name
LOGGER_NAME = "llm_utils"

# Default logging formats
# Concise format with date and time: MM-DD HH:MM:SS
DEFAULT_FORMAT = "%(asctime)s %(name)s - %(message)s"
# Detailed format with full timestamp
DETAILED_FORMAT = "%(asctime)s %(name)s:%(funcName)s:%(lineno)d - %(message)s"
# Simple format without timestamp (for cases where timestamps aren't needed)
SIMPLE_FORMAT = "%(name)s - %(message)s"

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
    """Set up logging configuration for the entire application.
    
    This function configures logging for ALL modules in the application,
    not just llm_utils. It should be called once at the start of your application.
    
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
        logging.Logger: The configured root logger instance.
    
    Example:
        >>> # Basic setup with timestamps (default)
        >>> logger = setup_logging()
        >>> # Output: 12-06 14:30:45 module_name - Message
        
        >>> # Debug level with detailed format
        >>> logger = setup_logging(level="DEBUG", use_detailed_format=True)
        >>> # Output: 12-06 14:30:45 [DEBUG] module_name:func:42 - Message
        
        >>> # Simple format without timestamps
        >>> logger = setup_logging(no_timestamps=True)
        >>> # Output: module_name - Message
        
        >>> # Custom date format with full year
        >>> logger = setup_logging(date_format="%Y-%m-%d %H:%M:%S")
        >>> # Output: 2025-12-06 14:30:45 module_name - Message
    """
    # Get the root logger to configure ALL modules
    root_logger = logging.getLogger()
    
    # Clear any existing handlers from root logger
    root_logger.handlers.clear()
    
    # Set the logging level on root logger
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
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
    
    # Create rich handler with console output
    console = Console(file=stream or sys.stdout, force_terminal=True)
    handler = RichHandler(
        console=console,
        show_time=not no_timestamps,
        show_level=True,
        show_path=use_detailed_format,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False
    )
    handler.setLevel(log_level)
    # Note: RichHandler formats its own output, but we set formatter for consistency
    handler.setFormatter(formatter)
    
    # Add handler to root logger (applies to all modules)
    root_logger.addHandler(handler)
    
    # Also configure the package logger for backward compatibility
    package_logger = logging.getLogger(LOGGER_NAME)
    package_logger.setLevel(log_level)
    # Package logger will inherit the root logger's handler
    package_logger.propagate = True
    
    # Set httpx and httpcore loggers to WARNING to reduce noise
    # These libraries log HTTP requests at INFO level by default
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    This returns a properly configured logger that inherits settings from
    the root logger configured by setup_logging().
    
    Args:
        name (Optional[str]): Module name to append to the package logger name.
            If None, returns the package-level logger.
    
    Returns:
        logging.Logger: Logger instance for the specified module.
    
    Example:
        >>> logger = get_logger("online_client")
        >>> logger.info("Client initialized")
        12-06 14:30:45 llm_utils.online_client - Client initialized
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


# Auto-setup with default configuration when package is imported
# Users can call setup_logging() again to customize settings
_default_logger = setup_logging()
