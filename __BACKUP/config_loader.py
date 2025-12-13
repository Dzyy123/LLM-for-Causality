"""
Causal Prompts Configuration Loader

Provides a centralized module for loading and accessing causal inference system
configuration from YAML files, including expert definitions, routing rules, and
prompt templates.
"""

import yaml
import os
from typing import Dict, List, Any, Optional


class CausalPromptsConfig:
    """
    Configuration loader and accessor for causal inference prompts and settings.
    
    This class provides convenient access to:
    - Expert definitions (capabilities, specialties, reasoning styles)
    - Routing rules (which experts to use for different question types)
    - Prompt templates (for expert analysis and distractor generation)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file.
                        If None, uses default 'causal_prompts.yaml' in current directory.
        """
        if config_path is None:
            # Default to causal_prompts.yaml in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'causal_prompts.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please ensure 'causal_prompts.yaml' exists in the project directory."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    # ===== Expert Configuration =====
    
    def get_expert_config(self, expert_type: str) -> Dict[str, str]:
        """
        Get configuration for a specific expert type.
        
        Args:
            expert_type: Expert identifier (e.g., 'graph_theory', 'statistical')
            
        Returns:
            Dictionary with expert configuration (name, description, specialty, etc.)
        """
        experts = self.config.get('experts', {})
        if expert_type not in experts:
            raise ValueError(
                f"Unknown expert type: {expert_type}\n"
                f"Available experts: {', '.join(experts.keys())}"
            )
        return experts[expert_type]
    
    def get_all_experts(self) -> Dict[str, Dict[str, str]]:
        """Get all expert configurations."""
        return self.config.get('experts', {})
    
    def get_expert_names(self) -> List[str]:
        """Get list of all available expert type names."""
        return list(self.config.get('experts', {}).keys())
    
    # ===== Routing Rules =====
    
    def get_routing_rule(self, question_type: str) -> List[str]:
        """
        Get recommended experts for a specific question type.
        
        Args:
            question_type: Type of causal question (e.g., 'backdoor_path', 'independence')
            
        Returns:
            List of expert types recommended for this question type
        """
        rules = self.config.get('routing_rules', {})
        if question_type not in rules:
            raise ValueError(
                f"Unknown question type: {question_type}\n"
                f"Available types: {', '.join(rules.keys())}"
            )
        return rules[question_type]
    
    def get_all_routing_rules(self) -> Dict[str, List[str]]:
        """Get all routing rules."""
        return self.config.get('routing_rules', {})
    
    # ===== Prompt Templates =====
    
    def get_prompt_template(self, prompt_type: str) -> str:
        """
        Get a causal analysis prompt template.
        
        Args:
            prompt_type: Type of prompt (e.g., 'expert_intro', 'backdoor_path', 'independence')
            
        Returns:
            Prompt template string (may contain format placeholders)
        """
        prompts = self.config.get('prompts', {})
        if prompt_type not in prompts:
            raise ValueError(
                f"Unknown prompt type: {prompt_type}\n"
                f"Available prompts: {', '.join(prompts.keys())}"
            )
        return prompts[prompt_type]
    
    def get_all_prompts(self) -> Dict[str, str]:
        """Get all causal analysis prompt templates."""
        return self.config.get('prompts', {})
    
    def format_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        Get and format a prompt template with provided values.
        
        Args:
            prompt_type: Type of prompt template
            **kwargs: Values to substitute into the template
            
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt_template(prompt_type)
        return template.format(**kwargs)
    
    # ===== Distractor Prompts =====
    
    def get_distractor_prompt(self, distractor_type: str) -> str:
        """
        Get a distractor prompt template.
        
        Args:
            distractor_type: Type of distractor ('contrarian', 'deceiver', 'hater', 
                            'distracted_prompt_template')
            
        Returns:
            Distractor prompt template string
        """
        distractors = self.config.get('distractor_prompts', {})
        if distractor_type not in distractors:
            raise ValueError(
                f"Unknown distractor type: {distractor_type}\n"
                f"Available distractors: {', '.join(distractors.keys())}"
            )
        return distractors[distractor_type]
    
    def get_all_distractor_prompts(self) -> Dict[str, str]:
        """Get all distractor prompt templates."""
        return self.config.get('distractor_prompts', {})
    
    def format_distractor_prompt(self, distractor_type: str, **kwargs) -> str:
        """
        Get and format a distractor prompt template.
        
        Args:
            distractor_type: Type of distractor prompt
            **kwargs: Values to substitute into the template
            
        Returns:
            Formatted distractor prompt string
        """
        template = self.get_distractor_prompt(distractor_type)
        return template.format(**kwargs)
    
    # ===== Utility Methods =====
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self.config


# Global singleton instance for convenient access
_global_config: Optional[CausalPromptsConfig] = None


def get_config(config_path: Optional[str] = None, force_reload: bool = False) -> CausalPromptsConfig:
    """
    Get the global configuration instance.
    
    This function provides a singleton pattern for accessing the configuration.
    The first call loads the configuration, subsequent calls return the cached instance.
    
    Args:
        config_path: Path to configuration file (only used on first call or if force_reload=True)
        force_reload: If True, reload configuration even if already loaded
        
    Returns:
        CausalPromptsConfig instance
    """
    global _global_config
    
    if _global_config is None or force_reload:
        _global_config = CausalPromptsConfig(config_path)
    
    return _global_config


# Convenience functions for direct access

def get_expert_config(expert_type: str) -> Dict[str, str]:
    """Get expert configuration (convenience function)."""
    return get_config().get_expert_config(expert_type)


def get_prompt_template(prompt_type: str) -> str:
    """Get causal analysis prompt template (convenience function)."""
    return get_config().get_prompt_template(prompt_type)


def get_distractor_prompt(distractor_type: str) -> str:
    """Get distractor prompt template (convenience function)."""
    return get_config().get_distractor_prompt(distractor_type)


def get_routing_rule(question_type: str) -> List[str]:
    """Get routing rule for question type (convenience function)."""
    return get_config().get_routing_rule(question_type)


def format_prompt(prompt_type: str, **kwargs) -> str:
    """Format a causal analysis prompt template (convenience function)."""
    return get_config().format_prompt(prompt_type, **kwargs)


def format_distractor_prompt(distractor_type: str, **kwargs) -> str:
    """Format a distractor prompt template (convenience function)."""
    return get_config().format_distractor_prompt(distractor_type, **kwargs)
