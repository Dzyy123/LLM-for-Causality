"""
Tree Query Package for Causal Discovery

This package provides a comprehensive framework for causal discovery using:
- Tree-based causal inference logic
- MoE (Mixture of Experts) architecture
- Adversarial confidence estimation
- Multi-calibration for probability adjustment

Example:
    >>> from tree_query import CausalDiscoveryFramework
    >>> from llm_utils import OnlineLLMClient
    >>> 
    >>> client = OnlineLLMClient(api_key="your-key", model_name="gpt-4")
    >>> framework = CausalDiscoveryFramework(
    ...     client=client,
    ...     all_variables=["X", "Y", "Z"]
    ... )
    >>> results = framework.tree_query("X", "Y")

:author: LLM-for-Causality Team
:date: 2025
"""

# Core framework
from tree_query.causal_discovery_framework import CausalDiscoveryFramework

# Experts
from tree_query.experts import (
    BaseExpert,
    BackdoorPathExpert,
    IndependenceExpert,
    LatentConfounderExpert,
    CausalDirectionExpert,
)

# Expert router
from tree_query.expert_router import ExpertRouter

# Confidence estimation
from tree_query.adversarial_confidence_estimator import (
    AdversarialConfidenceEstimator,
    create_adversarial_confidence_estimator,
)

# Configuration
from tree_query.config_loader import (
    CausalPromptsConfig,
    get_config,
    get_expert_config,
    get_prompt_template,
    get_adversarial_prompt,
    get_routing_rule,
    format_prompt,
    format_adversarial_prompt,
)

# Utilities
from tree_query.utils import (
    load_config,
    extract_yes_no_from_response,
    aggregate_expert_results,
)

__all__ = [
    # Core framework
    "CausalDiscoveryFramework",
    # Experts
    "BaseExpert",
    "BackdoorPathExpert",
    "IndependenceExpert",
    "LatentConfounderExpert",
    "CausalDirectionExpert",
    # Router
    "ExpertRouter",
    # Confidence
    "AdversarialConfidenceEstimator",
    "create_adversarial_confidence_estimator",
    # Config
    "CausalPromptsConfig",
    "get_config",
    "get_expert_config",
    "get_prompt_template",
    "get_adversarial_prompt",
    "get_routing_rule",
    "format_prompt",
    "format_adversarial_prompt",
    # Utils
    "load_config",
    "extract_yes_no_from_response",
    "aggregate_expert_results",
]
