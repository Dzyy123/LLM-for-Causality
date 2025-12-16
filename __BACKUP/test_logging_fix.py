"""
Test script to verify logging configuration works across all modules.
"""

import logging
from llm_utils import setup_logging

# Setup logging once at application start
print("Setting up logging...")
setup_logging(level="INFO")

# Test 1: Module-level logger (like in causal_discovery_framework.py)
logger1 = logging.getLogger("causal_discovery_framework")
logger1.info("Test message from causal_discovery_framework")

# Test 2: Another module-level logger (like in distractor_confidence_estimator.py)
logger2 = logging.getLogger("distractor_confidence_estimator")
logger2.info("Test message from distractor_confidence_estimator")

# Test 3: llm_utils package logger
logger3 = logging.getLogger("llm_utils")
logger3.info("Test message from llm_utils")

# Test 4: Submodule logger
logger4 = logging.getLogger("llm_utils.online_client")
logger4.info("Test message from llm_utils.online_client")

# Test 5: Different log levels
logger1.debug("This debug message should NOT appear (level=INFO)")
logger1.warning("This warning message SHOULD appear")
logger1.error("This error message SHOULD appear")

print("\n=== Testing with detailed format ===")
setup_logging(level="DEBUG", use_detailed_format=True)

logger1.debug("Debug message with detailed format")
logger1.info("Info message with detailed format")

print("\n=== All logging tests completed ===")
print("If you see timestamps and module names in the format:")
print("  MM-DD HH:MM:SS [LEVEL] module_name - message")
print("Then logging is working correctly for all modules!")
