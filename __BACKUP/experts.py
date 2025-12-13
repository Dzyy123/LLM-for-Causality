"""
Expert Classes Module

Defines the base expert class and all specialized expert implementations.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
from llm_utils import LocalLLMClient, OnlineLLMClient
from utils import load_config, extract_yes_no_from_response


# Load configurations
_CONFIG = load_config()
EXPERT_CONFIGS = _CONFIG['experts']
PROMPT_TEMPLATES = _CONFIG['prompts']


@dataclass
class BaseExpert(ABC):
    """
    Base class for all causal experts.
    
    Attributes:
        base_prompt: The base question prompt
        x1: First variable name
        x2: Second variable name
        client: LLM client (LocalLLMClient or OnlineLLMClient)
        all_variables: List of all variables in the causal system
        expert_type: Type identifier for the expert
        seed: Random seed for reproducible LLM responses (optional)
    """
    base_prompt: str
    x1: str
    x2: str
    client: Union[LocalLLMClient, OnlineLLMClient]
    all_variables: List[str]
    expert_type: str = "base"
    seed: Optional[int] = None
    
    def get_expert_config(self) -> Dict[str, str]:
        """Get the configuration for this expert type."""
        return EXPERT_CONFIGS.get(self.expert_type, {})
    
    @abstractmethod
    def generate_question(self) -> str:
        """Generate the specific question for this expert."""
        pass
    
    def create_expert_prompt(self) -> str:
        """Create a specialized prompt incorporating expert perspective."""
        expert_config = self.get_expert_config()
        if not expert_config:
            return self.base_prompt
        
        expert_intro = PROMPT_TEMPLATES['expert_intro'].format(**expert_config)
        return expert_intro + "\n" + self.base_prompt
    
    def judge(self) -> Dict[str, Any]:
        """
        Execute the expert's judgment.
        
        Returns:
            Dictionary containing:
                - label: 1 (Yes) or 0 (No)
                - expert: Expert type identifier
        """
        prompt = self.create_expert_prompt()
        return self._judge_prompt(prompt)
    
    def judge_with_confidence(
        self,
        confidence_estimator=None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the expert's judgment with confidence estimation.
        
        Args:
            confidence_estimator: ConfidenceEstimator instance for computing confidence
            verbose: Whether to print detailed confidence estimation information
            
        Returns:
            Dictionary containing:
                - label: 1 (Yes) or 0 (No)
                - expert: Expert type identifier
                - confidence: Confidence metrics (if estimator provided)
        """
        prompt = self.create_expert_prompt()
        result = self._judge_prompt(prompt)
        
        # Add confidence estimation if estimator provided
        if confidence_estimator is not None:
            confidence = self._compute_confidence(
                result['label'],
                confidence_estimator,
                verbose
            )
            result['confidence'] = confidence
        
        return result
    
    def _judge_prompt(self, prompt: str) -> Dict[str, Any]:
        """Execute basic judgment by calling the model."""
        try:
            response = self.client.chat(prompt, temperature=0.7, seed=self.seed)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract label from response
            label = extract_yes_no_from_response(response_text)
            
            return {
                "label": label,
                "expert": self.expert_type,
                "response": response_text  # Include the full response text
            }
        except Exception as e:
            print(f"Judgment failed: {e}")
            return {
                "label": 0,
                "expert": self.expert_type,
                "response": f"Error: {e}"  # Include error message
            }
    
    def _compute_confidence(
        self,
        label: int,
        confidence_estimator,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Compute confidence scores for the judgment using the provided estimator.
        
        Args:
            label: The judgment result (0 or 1)
            confidence_estimator: ConfidenceEstimator instance
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with confidence metrics from the estimator
        """
        return confidence_estimator.estimate_confidence(
            expert=self,
            result=label,
            verbose=verbose
        )


@dataclass
class BackdoorPathExpert(BaseExpert):
    """Expert for checking backdoor paths between variables."""
    
    def generate_question(self) -> str:
        """Generate question about backdoor paths."""
        return PROMPT_TEMPLATES['backdoor_path'].format(
            all_variables=', '.join(self.all_variables),
            x1=self.x1,
            x2=self.x2
        )


@dataclass
class IndependenceExpert(BaseExpert):
    """Expert for checking independence between variables."""
    
    after_blocking: bool = False
    
    def generate_question(self) -> str:
        """Generate question about independence."""
        template_key = 'independence_after_blocking' if self.after_blocking else 'independence'
        return PROMPT_TEMPLATES[template_key].format(
            all_variables=', '.join(self.all_variables),
            x1=self.x1,
            x2=self.x2
        )


@dataclass
class LatentConfounderExpert(BaseExpert):
    """Expert for checking latent confounders."""
    
    after_blocking: bool = False
    
    def generate_question(self) -> str:
        """Generate question about latent confounders."""
        template_key = 'latent_confounder_after_blocking' if self.after_blocking else 'latent_confounder'
        return PROMPT_TEMPLATES[template_key].format(
            all_variables=', '.join(self.all_variables),
            x1=self.x1,
            x2=self.x2
        )


@dataclass
class CausalDirectionExpert(BaseExpert):
    """Expert for determining causal direction."""
    
    after_blocking: bool = False
    
    def generate_question(self) -> str:
        """Generate question about causal direction."""
        template_key = 'causal_direction_after_blocking' if self.after_blocking else 'causal_direction'
        return PROMPT_TEMPLATES[template_key].format(
            all_variables=', '.join(self.all_variables),
            x1=self.x1,
            x2=self.x2
        )
