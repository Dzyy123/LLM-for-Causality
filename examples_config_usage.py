"""
Configuration Loader Usage Examples

Demonstrates how to use the new config_loader module in different scenarios.
"""

# ==============================================================================
# Example 1: Basic Configuration Access
# ==============================================================================

def example_basic_access():
    """Basic configuration loading and access."""
    from tree_query import get_config
    
    # Get configuration instance (loaded once, cached globally)
    config = get_config()
    
    # Access expert configuration
    expert = config.get_expert_config('domain_knowledge')
    print(f"Expert: {expert['name']}")
    print(f"Specialty: {expert['specialty']}\n")
    
    # List all available experts
    experts = config.get_expert_names()
    print(f"Available experts: {', '.join(experts)}\n")


# ==============================================================================
# Example 2: Using Convenience Functions
# ==============================================================================

def example_convenience_functions():
    """Use convenience functions for quick access."""
    from tree_query.config_loader import (
        get_expert_config,
        get_routing_rule,
        format_prompt
    )
    
    # Get expert config directly
    expert = get_expert_config('statistical')
    print(f"Expert: {expert['name']}\n")
    
    # Get routing rule
    recommended_experts = get_routing_rule('causal_direction')
    print(f"Experts for causal_direction: {', '.join(recommended_experts[:3])}\n")
    
    # Format a prompt
    prompt = format_prompt(
        'independence',
        all_variables=['X', 'Y', 'Z'],
        x1='X',
        x2='Y'
    )
    print(f"Formatted prompt:\n{prompt[:200]}...\n")


# ==============================================================================
# Example 3: Distractor Prompt Integration
# ==============================================================================

def example_distractor_prompts():
    """Working with distractor confidence estimator prompts."""
    from tree_query.config_loader import format_distractor_prompt, get_config
    
    # Format a contrarian distractor prompt
    contrarian = format_distractor_prompt(
        'contrarian',
        question='Does smoking cause lung cancer?',
        answer='Yes, smoking is a well-established cause of lung cancer...',
        conclusion='Yes',
        opposite_conclusion='No'
    )
    
    print("Contrarian distractor prompt:")
    print(contrarian[:300])
    print("...\n")
    
    # Get the distracted prompt template
    config = get_config()
    template = config.get_distractor_prompt('distracted_prompt_template')
    print(f"Template length: {len(template)} characters\n")


# ==============================================================================
# Example 4: Expert System Integration
# ==============================================================================

def example_expert_system_integration():
    """How to use config in an expert system."""
    from tree_query.config_loader import get_expert_config, format_prompt
    
    # Simulate expert initialization
    expert_type = 'domain_knowledge'
    expert_config = get_expert_config(expert_type)
    
    print(f"Initializing {expert_config['name']}...")
    print(f"Reasoning style: {expert_config['reasoning_style']}")
    
    # Create a causal direction question
    question = format_prompt(
        'causal_direction',
        all_variables=['Smoking', 'Lung Cancer', 'Age'],
        x1='Smoking',
        x2='Lung Cancer'
    )
    
    print(f"\nGenerated question:\n{question}\n")


# ==============================================================================
# Example 5: Iterating Over Configuration
# ==============================================================================

def example_iteration():
    """Iterate over configuration elements."""
    from tree_query import get_config
    
    config = get_config()
    
    # Iterate over all experts
    print("All Expert Types:")
    all_experts = config.get_all_experts()
    for expert_type, expert_info in all_experts.items():
        print(f"  - {expert_type}: {expert_info['name']}")
    
    print("\nAll Routing Rules:")
    all_rules = config.get_all_routing_rules()
    for question_type, experts in all_rules.items():
        print(f"  - {question_type}: {len(experts)} experts")
    
    print("\nAll Distractor Prompts:")
    all_distractors = config.get_all_distractor_prompts()
    for distractor_type, prompt in all_distractors.items():
        print(f"  - {distractor_type}: {len(prompt)} characters")


# ==============================================================================
# Example 6: Error Handling
# ==============================================================================

def example_error_handling():
    """Demonstrate error handling."""
    from tree_query.config_loader import get_expert_config, get_prompt_template
    
    try:
        # Try to access non-existent expert
        expert = get_expert_config('nonexistent_expert')
    except ValueError as e:
        print(f"Error caught: {e}\n")
    
    try:
        # Try to access non-existent prompt
        prompt = get_prompt_template('nonexistent_prompt')
    except ValueError as e:
        print(f"Error caught: {e}\n")


# ==============================================================================
# Example 7: Custom Configuration Path
# ==============================================================================

def example_custom_path():
    """Load configuration from custom path."""
    from tree_query import get_config
    
    # Load from custom path (if you have multiple config files)
    # config = get_config(config_path='/path/to/custom_config.yaml', force_reload=True)
    
    # Or use default path
    config = get_config()
    print(f"Configuration loaded from: {config.config_path}\n")


# ==============================================================================
# Main Demo
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Configuration Loader Usage Examples")
    print("="*80 + "\n")
    
    print("\n" + "-"*80)
    print("Example 1: Basic Configuration Access")
    print("-"*80)
    example_basic_access()
    
    print("\n" + "-"*80)
    print("Example 2: Using Convenience Functions")
    print("-"*80)
    example_convenience_functions()
    
    print("\n" + "-"*80)
    print("Example 3: Distractor Prompt Integration")
    print("-"*80)
    example_distractor_prompts()
    
    print("\n" + "-"*80)
    print("Example 4: Expert System Integration")
    print("-"*80)
    example_expert_system_integration()
    
    print("\n" + "-"*80)
    print("Example 5: Iterating Over Configuration")
    print("-"*80)
    example_iteration()
    
    print("\n" + "-"*80)
    print("Example 6: Error Handling")
    print("-"*80)
    example_error_handling()
    
    print("\n" + "-"*80)
    print("Example 7: Custom Configuration Path")
    print("-"*80)
    example_custom_path()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
