"""
Expert Router Module

Handles intelligent routing and selection of experts for different question types.
"""

from typing import List, Union
from llm_utils import LocalLLMClient, OnlineLLMClient
from utils import load_config


# Load configurations
_CONFIG = load_config()
EXPERT_CONFIGS = _CONFIG['experts']
ROUTING_RULES = _CONFIG['routing_rules']
PROMPT_TEMPLATES = _CONFIG['prompts']


class ExpertRouter:
    """Routes questions to appropriate experts based on question type."""
    
    def __init__(self, client: Union[LocalLLMClient, OnlineLLMClient]):
        """
        Initialize the router.
        
        Args:
            client: LLM client for the clinic agent
        """
        self.clinic_client = client
    
    def select_experts(
        self,
        question_type: str,
        x1: str,
        x2: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Select the most relevant experts for a given question.
        
        Args:
            question_type: Type of question (e.g., 'backdoor_path', 'independence')
            x1: First variable name
            x2: Second variable name
            top_k: Number of experts to select
            
        Returns:
            List of selected expert type identifiers
        """
        # Get base experts from routing rules
        base_experts = ROUTING_RULES.get(question_type, list(EXPERT_CONFIGS.keys()))
        
        # Try intelligent selection using clinic agent
        try:
            recommended_experts = self._clinic_agent_recommend(
                question_type, x1, x2, base_experts
            )
            return recommended_experts[:top_k]
        except Exception as e:
            print(f"Clinic agent routing failed: {e}, using base routing")
            return base_experts[:top_k]
    
    def _clinic_agent_recommend(
        self,
        question_type: str,
        x1: str,
        x2: str,
        base_experts: List[str]
    ) -> List[str]:
        """Use clinic LLM agent to recommend experts based on variable context."""
        experts_description = "\n".join([
            f"- {expert}: {EXPERT_CONFIGS[expert]['description']}"
            for expert in base_experts
        ])
        
        clinic_prompt = PROMPT_TEMPLATES['clinic_router'].format(
            question_type=question_type,
            x1=x1,
            x2=x2,
            experts_description=experts_description
        )
        
        response = self.clinic_client.chat(clinic_prompt)
        response_text = response.content.strip()
        
        recommended_experts = self._parse_clinic_recommendation(response_text, base_experts)
        return recommended_experts
    
    def _parse_clinic_recommendation(
        self,
        response_text: str,
        base_experts: List[str]
    ) -> List[str]:
        """Parse the clinic agent's recommendation."""
        # Method 1: Look for "Final Recommended Experts" in text
        if "Final Recommended Experts" in response_text or "final recommended experts" in response_text.lower():
            parts = response_text.lower().split("final recommended experts")
            if len(parts) > 1:
                expert_line = parts[1].strip().lstrip(":").strip()
                experts = self._extract_experts_from_line(expert_line, base_experts)
                if len(experts) >= 2:
                    return experts
        
        # Method 2: Check last line
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            experts = self._extract_experts_from_line(last_line, base_experts)
            if len(experts) >= 2:
                return experts
        
        # Method 3: Search for expert names in entire text
        found_experts = []
        for expert in base_experts:
            if expert in response_text:
                found_experts.append(expert)
        
        if len(found_experts) >= 2:
            return found_experts[:3]
        
        # Fallback: return first 3 base experts
        return base_experts[:3]
    
    def _extract_experts_from_line(
        self,
        line: str,
        base_experts: List[str]
    ) -> List[str]:
        """Extract expert names from a line of text."""
        experts = []
        
        # Clean the line
        clean_line = line.replace(' ', '')
        
        # Try different separators
        separators = [',', ';']
        parts = [clean_line]
        
        for sep in separators:
            if sep in clean_line:
                parts = [part.strip() for part in clean_line.split(sep)]
                break
        
        for part in parts:
            clean_part = part.lower().replace('expert', '').strip()
            
            # Match expert names
            for expert in base_experts:
                if (expert in clean_part or
                    expert.replace('_', ' ') in clean_part or
                    EXPERT_CONFIGS[expert]['name'].lower() in part.lower()):
                    if expert not in experts:
                        experts.append(expert)
                        break
            
            if len(experts) >= 3:
                break
        
        return experts
