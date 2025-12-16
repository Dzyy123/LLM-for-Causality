"""
Causal Expert System with Mixture of Experts (MoE) Architecture

Main interface module for the causal inference system using multiple specialized experts
that analyze causal relationships from different perspectives.
"""

from typing import List, Dict, Any, Tuple, Union
from llm_utils import LocalLLMClient, OnlineLLMClient, setup_logging
from causal_tasks import TreeQueryTask, PairwiseCausalAnalysisTask


# ============== Example Usage ==============

if __name__ == "__main__":
    # Setup logging
    setup_logging(level="INFO")
    
    # Create LLM client
    # client = LocalLLMClient(
    #     model_id="Qwen/Qwen3-4B-Thinking-2507",
    #     local_path="./models/qwen3-4b-thinking",
    #     device="auto",
    #     max_tokens=200,
    #     temperature=0.7
    # )

    # Create Online LLM client
    client = OnlineLLMClient(
        api_key="sk-fnUHDzxXAimEnYgyX20Jag",
        base_url="https://llmapi.paratera.com/v1/",
        model_name="Qwen3-Next-80B-A3B-Thinking",
        max_tokens=800,
        temperature=0.7
    )
    
    # Define variables
    all_variables = ["Ice Cream Sales", "Drowning Incidents", "Temperature"]
    
    print("Starting Causal Inference Analysis")
    print("=" * 60)
    
    # Create expert system
    expert_system = TreeQueryTask(
        client,
        all_variables
    )
    
    # Query causal relationship
    result = expert_system.query("Ice Cream Sales", "Drowning Incidents")
    
    print(f"\n{'='*60}")
    print("=== Final Result ===")
    print(f"Causal Relationship: {result['relation']}")
    
    print(f"\n=== Detailed Execution Log ===")
    for step_name, step_result in result["log"]:
        print(f"\n{step_name}:")
        print(f"  Final Judgment: {'Yes' if step_result['label'] == 1 else 'No'}")
        
        if "expert_results" in step_result:
            print(f"  Expert Details:")
            for expert_result in step_result["expert_results"]:
                expert_name = expert_result.get("expert", "unknown")
                print(f"    - {expert_name}: {'Yes' if expert_result['label'] == 1 else 'No'}")
    
    print("\n" + "=" * 60)
    
    # Clean up
    if hasattr(client, 'unload_model'):
        client.unload_model()
        print("Model unloaded")
