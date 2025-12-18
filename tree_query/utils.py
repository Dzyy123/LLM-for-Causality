"""
Utility Functions and Configuration Loading

Common utility functions and configuration management for the causal expert system.
"""

import yaml
import os
from typing import Dict, Any, List


# ============== Configuration Loading ==============

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses default causal_prompts.yaml
        
    Returns:
        Dictionary containing the configuration with keys: experts, routing_rules, prompts
        
    Note:
        This function is kept for backward compatibility.
        New code should use config_loader.get_config() instead.
    """
    if config_path is None:
        # Default to causal_prompts.yaml in the tree_query directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'causal_prompts.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============== Response Processing ==============

def extract_yes_no_from_response(response_text: str) -> int:
    """
    Extract Yes/No label from model response text.
    Only checks the first line of the response.
    
    Args:
        response_text: The response text from the model
        
    Returns:
        1 for Yes, 0 for No
    """
    # Only check the first line
    first_line = response_text.split('\n')[0].lower().strip()
    
    # Simple heuristic: check if yes appears before no
    yes_pos = first_line.find('yes')
    no_pos = first_line.find('no')
    
    if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
        return 1
    elif no_pos >= 0:
        return 0
    else:
        # Default to No if unclear
        return 0


# ============== Expert Aggregation ==============

def aggregate_expert_results(expert_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple expert judgments into a final decision using majority voting.
    
    Args:
        expert_results: List of expert judgment results
        
    Returns:
        Aggregated result with label and expert details
    """
    if not expert_results:
        return {"label": 0}
    
    # Simple majority voting
    yes_votes = sum(1 for result in expert_results if result["label"] == 1)
    total_votes = len(expert_results)
    
    # Determine final label by majority
    final_label = 1 if yes_votes > total_votes / 2 else 0
    
    return {
        "label": final_label,
        "expert_results": expert_results
    }
