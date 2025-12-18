"""
Adversarial Confidence Estimator Module (Version 2 - Simplified)

Version 2: Uses only Yes/No conclusion (without full explanation) in adversarially-influenced sampling prompts.
This version differs from v1 by providing only the conclusion
(not the complete original answer text) when presenting adversarial arguments for re-evaluation.

Uses multi-threading for parallel LLM requests and modular design.
"""

import math
import random
import logging
from typing import List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from llm_utils import LocalLLMClient, OnlineLLMClient
from tree_query import extract_yes_no_from_response, get_config


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class OriginalAnswer:
    """Container for original answer data."""
    text: str
    label: int
    index: int


@dataclass
class AdversaryArgument:
    """Container for a single adversary argument."""
    dtype: str
    prompt: str
    argument: str


@dataclass
class AdversarySet:
    """Container for a complete set of adversary arguments."""
    set_index: int
    original_answer: str
    original_label: int
    contrarian: AdversaryArgument
    deceiver: AdversaryArgument
    hater: AdversaryArgument


@dataclass
class AdversarySample:
    """Container for an adversarially-influenced sample result."""
    set_index: int
    adversarial_type: str
    original_label: int
    sampling_prompt: str
    response: str
    label: int


class SeedManager:
    """Manages seed generation for reproducibility."""
    
    def __init__(self, seed: int = None):
        self.use_seed = seed is not None
        self.current_seed = seed
    
    def get_next(self) -> Union[int, None]:
        """Get next seed value."""
        if not self.use_seed:
            return random.randint(0, int(1e6))
        current = self.current_seed
        self.current_seed += 1
        return current


class PromptBuilder:
    """Builds prompts for adversarial generation and sampling."""
    
    def __init__(self):
        config = get_config()
        self.contrarian_template = config.get_adversarial_prompt('contrarian')
        self.deceiver_template = config.get_adversarial_prompt('deceiver')
        self.hater_template = config.get_adversarial_prompt('hater')
        self.single_contrarian_template = config.get_adversarial_prompt('single_contrarian_template')
        self.single_deceiver_template = config.get_adversarial_prompt('single_deceiver_template')
        self.single_hater_template = config.get_adversarial_prompt('single_hater_template')
    
    def build_generation_prompt(self, dtype: str, question: str, 
                                answer: str, conclusion: str, 
                                opposite_conclusion: str) -> str:
        """Build prompt for generating adversarial argument."""
        template_map = {
            'contrarian': self.contrarian_template,
            'deceiver': self.deceiver_template,
            'hater': self.hater_template
        }
        template = template_map[dtype]
        return template.format(
            question=question,
            answer=answer,
            conclusion=conclusion,
            opposite_conclusion=opposite_conclusion
        )
    
    def build_sampling_prompt(self, dtype: str, question: str, 
                             original_label: int, 
                             adversarial_argument: str) -> str:
        """Build prompt for sampling with adversarial argument."""
        template_map = {
            'contrarian': self.single_contrarian_template,
            'deceiver': self.single_deceiver_template,
            'hater': self.single_hater_template
        }
        template = template_map[dtype]
        original_conclusion = "Yes" if original_label == 1 else "No"
        return template.format(
            question=question,
            separator="",
            adversarial_argument=adversarial_argument,
            original_conclusion=original_conclusion
        )


class LLMRequestExecutor:
    """Executes LLM requests with threading support."""
    
    def __init__(self, client: Union[LocalLLMClient, OnlineLLMClient], 
                 seed_manager: SeedManager, max_workers: int = 10):
        self.client = client
        self.seed_manager = seed_manager
        self.max_workers = max_workers
    
    def execute_single(self, prompt: str) -> str:
        """Execute single LLM request."""
        try:
            response = self.client.chat(prompt, seed=self.seed_manager.get_next())
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"LLM request failed: {e}")
            return "[Error: Request failed]"
    
    def execute_parallel(self, prompts: List[Tuple[Any, str]], 
                        max_workers: int = None) -> List[Tuple[Any, str]]:
        """Execute multiple LLM requests in parallel."""
        results = []
        workers = max_workers if max_workers is not None else self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {
                executor.submit(self.execute_single, prompt): key 
                for key, prompt in prompts
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                result = future.result()
                results.append((key, result))
        return results


def sample_original_answers(executor: LLMRequestExecutor, question: str, k_samples: int) -> List[OriginalAnswer]:
    """Sample k original answers in parallel."""
    logger.info(f"Sampling {k_samples} original answers")
    
    prompts = [(i, question) for i in range(k_samples)]
    results = executor.execute_parallel(prompts)
    
    answers = []
    for idx, text in sorted(results):
        label = extract_yes_no_from_response(text)
        answers.append(OriginalAnswer(text=text, label=label, index=idx))
    
    logger.info(f"Completed {len(answers)} original answers")
    return answers


class AdversaryGenerator:
    """Generates adversarial arguments."""
    
    def __init__(self, executor: LLMRequestExecutor, prompt_builder: PromptBuilder, max_workers: int = 10):
        self.executor = executor
        self.prompt_builder = prompt_builder
        self.max_workers = max_workers
    
    def generate_single_set(self, question: str, original_answer: str, 
                           original_label: int) -> Tuple[AdversaryArgument, AdversaryArgument, AdversaryArgument]:
        """Generate one complete set of three adversarial arguments in parallel."""
        conclusion = "Yes" if original_label == 1 else "No"
        opposite_conclusion = "No" if original_label == 1 else "Yes"
        
        # Build all three generation prompts
        prompts = []
        for dtype in ['contrarian', 'deceiver', 'hater']:
            prompt = self.prompt_builder.build_generation_prompt(
                dtype, question, original_answer, conclusion, opposite_conclusion
            )
            prompts.append((dtype, prompt))
        
        # Execute in parallel
        results = self.executor.execute_parallel(prompts, max_workers=3)
        
        # Build AdversaryArgument objects
        adversarial_map = {}
        for dtype, argument in results:
            # Find the prompt used
            prompt = next(p for dt, p in prompts if dt == dtype)
            adversarial_map[dtype] = AdversaryArgument(
                dtype=dtype,
                prompt=prompt,
                argument=argument
            )
        
        return (adversarial_map['contrarian'], 
                adversarial_map['deceiver'], 
                adversarial_map['hater'])
    
    def generate_all_sets(self, question: str, 
                         original_answers: List[OriginalAnswer],
                         k2_samples: int) -> List[AdversarySet]:
        """Generate all adversarial sets (k1 × k2 sets) in parallel."""
        total_sets = len(original_answers) * k2_samples
        logger.info(f"Generating {total_sets} adversarial sets")
        
        # Build task list
        tasks = []
        set_index = 0
        for orig_answer in original_answers:
            for j in range(k2_samples):
                tasks.append((
                    set_index,
                    orig_answer.text,
                    orig_answer.label,
                    question
                ))
                set_index += 1
        
        # Execute with thread pool
        adversarial_sets = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.generate_single_set, 
                    task[3], task[1], task[2]
                ): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                set_idx, orig_text, orig_label, _ = task
                contrarian, deceiver, hater = future.result()
                
                adversarial_sets.append(AdversarySet(
                    set_index=set_idx,
                    original_answer=orig_text,
                    original_label=orig_label,
                    contrarian=contrarian,
                    deceiver=deceiver,
                    hater=hater
                ))
        
        # Sort by set_index to maintain order
        adversarial_sets.sort(key=lambda x: x.set_index)
        logger.info(f"Completed {len(adversarial_sets)} adversarial sets")
        return adversarial_sets


class AdversarySampler:
    """Samples answers after presenting adversarial arguments."""
    
    def __init__(self, executor: LLMRequestExecutor, prompt_builder: PromptBuilder, max_workers: int = 15):
        self.executor = executor
        self.prompt_builder = prompt_builder
        self.max_workers = max_workers
    
    def sample_with_adversarial(self, question: str, adversarial_set: AdversarySet,
                               dtype: str) -> AdversarySample:
        """Sample with one adversarial type."""
        adversarial = getattr(adversarial_set, dtype)
        
        # Build sampling prompt
        sampling_prompt = self.prompt_builder.build_sampling_prompt(
            dtype, question, adversarial_set.original_label, adversarial.argument
        )
        
        # Execute request
        response = self.executor.execute_single(sampling_prompt)
        label = extract_yes_no_from_response(response)
        
        return AdversarySample(
            set_index=adversarial_set.set_index,
            adversarial_type=dtype,
            original_label=adversarial_set.original_label,
            sampling_prompt=sampling_prompt,
            response=response,
            label=label
        )
    
    def sample_all(self, question: str, 
                   adversarial_sets: List[AdversarySet]) -> List[AdversarySample]:
        """Sample with all adversarial arguments (k1 × k2 × 3 samples) in parallel."""
        total_samples = len(adversarial_sets) * 3
        logger.info(f"Sampling {total_samples} adversarially-influenced responses")
        
        # Build task list
        tasks = []
        for aset in adversarial_sets:
            for dtype in ['contrarian', 'deceiver', 'hater']:
                tasks.append((question, aset, dtype))
        
        # Execute in parallel
        samples = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.sample_with_adversarial,
                    task[0], task[1], task[2]
                ): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                sample = future.result()
                samples.append(sample)
        
        # Sort by set_index and adversarial_type
        samples.sort(key=lambda x: (x.set_index, x.adversarial_type))
        logger.info(f"Completed {len(samples)} adversarially-influenced responses")
        return samples


class MetricsCalculator:
    """Calculates confidence metrics."""
    
    def __init__(self, gamma1: float = 1e-6, gamma2: float = 1e-6):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
    def compute_flip_rate(self, original_result: int, 
                         adversarial_results: List[int]) -> float:
        """Compute flip rate for one adversarial type."""
        if not adversarial_results:
            return 0.0
        flips = sum(1 for r in adversarial_results if r != original_result)
        return flips / len(adversarial_results)
    
    def compute_label_distribution(self, adversarial_results: List[int]) -> Dict[str, int]:
        """Compute Yes/No distribution."""
        yes_count = sum(1 for r in adversarial_results if r == 1)
        no_count = len(adversarial_results) - yes_count
        return {'yes': yes_count, 'no': no_count}
    
    def compute_confidence_score(self, p0: float, p1: float, 
                                 p2: float, p3: float) -> float:
        """Compute final confidence score."""
        if p0 == 0:
            return 0.0
        
        deviation_sum = (abs(p1 - p0) / p0 + abs(p2 - p0) / p0 + abs(p3 - p0) / p0)
        avg_relative_deviation = deviation_sum / 3.0
        confidence = p0 * (1 - avg_relative_deviation)
        
        return max(0.0, confidence)
    
    def calculate_all_metrics(self, expert_result: int,
                             adversarial_samples: List[AdversarySample]) -> Dict[str, Any]:
        """Calculate all confidence metrics."""
        # Build detailed tracking: group by set_index to show each original answer's journey
        detailed_tracking = {}
        for sample in adversarial_samples:
            if sample.set_index not in detailed_tracking:
                detailed_tracking[sample.set_index] = {
                    'original_label': sample.original_label,
                    'contrarian': None,
                    'deceiver': None,
                    'hater': None
                }
            detailed_tracking[sample.set_index][sample.adversarial_type] = sample.label
        
        # Calculate flip rates and label distributions based on EACH adversarial set's original label
        # Group samples by adversarial type for proper flip rate calculation
        flip_rates = {}
        label_distributions = {}
        
        for dtype in ['contrarian', 'deceiver', 'hater']:
            type_samples = [s for s in adversarial_samples if s.adversarial_type == dtype]
            
            # Calculate flip rate: compare each adversarially-influenced result with its OWN original label
            total_flips = sum(1 for s in type_samples if s.label != s.original_label)
            flip_rates[dtype] = total_flips / len(type_samples) if type_samples else 0.0
            
            # Label distribution
            label_distributions[dtype] = self.compute_label_distribution([s.label for s in type_samples])
        
        # Calculate probabilities
        p0 = 1.0
        p1 = 1 - flip_rates['contrarian']
        p2 = 1 - flip_rates['deceiver']
        p3 = 1 - flip_rates['hater']
        
        # Calculate confidence score
        confidence_score = self.compute_confidence_score(p0, p1, p2, p3)
        
        # Calculate averages
        avg_flip_rate = sum(flip_rates.values()) / 3
        robustness_score = 1 - avg_flip_rate
        
        return {
            'flip_rates': flip_rates,
            'label_distributions': label_distributions,
            'detailed_tracking': detailed_tracking,
            'avg_flip_rate': avg_flip_rate,
            'robustness_score': robustness_score,
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'confidence_score': confidence_score
        }


class AdversarialConfidenceEstimatorV2:
    """
    Main estimator class coordinating all components (Version 2 - Simplified).
    Uses multi-threading for parallel LLM requests.
    V2: Uses only Yes/No conclusion without full explanation in adversarially-influenced sampling.
    """
    
    def __init__(self, client: Union[LocalLLMClient, OnlineLLMClient],
                 k1_samples: int = 20, k2_samples: int = 1,
                 gamma1: float = 1e-6, gamma2: float = 1e-6,
                 seed: int = None, max_workers: int = 10):
        """
        Initialize the estimator.
        
        Args:
            client: LLM client for requests
            k1_samples: Number of original answer samples
            k2_samples: Number of adversarial sets per original answer
            gamma1: Stabilization parameter for numerator
            gamma2: Stabilization parameter for denominator
            seed: Random seed (auto-increments per request)
            max_workers: Maximum number of threads for parallel LLM requests
        """
        self.k1_samples = k1_samples
        self.k2_samples = k2_samples
        self.initial_seed = seed
        self.max_workers = max_workers
        
        # Initialize components
        self.seed_manager = SeedManager(seed)
        self.prompt_builder = PromptBuilder()
        self.executor = LLMRequestExecutor(client, self.seed_manager, max_workers)
        self.adversarial_generator = AdversaryGenerator(self.executor, self.prompt_builder, max_workers)
        self.adversarial_sampler = AdversarySampler(self.executor, self.prompt_builder, max_workers)
        self.metrics_calculator = MetricsCalculator(gamma1, gamma2)
    
    def estimate_confidence(self, expert, result: int) -> Dict[str, Any]:
        """
        Estimate confidence score with adversary-based testing (V2: Yes/No only).
        
        Args:
            expert: Expert instance with base_prompt
            result: Expert's answer (0 or 1)
        
        Returns:
            Dictionary with all metrics and intermediate data
        """
        question = expert.base_prompt
        
        logger.info("Starting confidence estimation (V2: Yes/No only)")
        logger.info(f"Expert: {expert.expert_type}, Result: {'Yes' if result == 1 else 'No'}")
        logger.info(f"Config: k1={self.k1_samples}, k2={self.k2_samples}, total={self.k1_samples * self.k2_samples * 3}")
        
        # Step 1: Sample original answers
        original_answers = sample_original_answers(self.executor, question, self.k1_samples)
        
        # Step 2: Generate adversarial sets
        adversarial_sets = self.adversarial_generator.generate_all_sets(
            question, original_answers, self.k2_samples
        )
        
        # Step 3: Sample with adversarial arguments
        adversarial_samples = self.adversarial_sampler.sample_all(
            question, adversarial_sets
        )
        
        # Step 4: Calculate metrics
        logger.info("Calculating metrics")
        metrics = self.metrics_calculator.calculate_all_metrics(
            result, adversarial_samples
        )
        
        logger.info(f"Confidence: {metrics['confidence_score']:.4f}, Robustness: {metrics['robustness_score']:.2%}")
        
        # Build comprehensive result
        return {
            **metrics,
            'result': result,
            'expert_type': expert.expert_type,
            'k1_samples': self.k1_samples,
            'k2_samples': self.k2_samples,
            'total_samples': self.k1_samples * self.k2_samples * 3,
            'original_answers': original_answers,
            'adversarial_sets': adversarial_sets,
            'adversarial_samples': adversarial_samples
        }


def create_adversarial_confidence_estimator_v2(
    client: Union[LocalLLMClient, OnlineLLMClient],
    k1_samples: int = 50, k2_samples: int = 1,
    gamma1: float = 1e-6, gamma2: float = 1e-6,
    seed: int = None, max_workers: int = 10
) -> AdversarialConfidenceEstimatorV2:
    """Factory function to create estimator V2."""
    return AdversarialConfidenceEstimatorV2(
        client=client,
        k1_samples=k1_samples,
        k2_samples=k2_samples,
        gamma1=gamma1,
        gamma2=gamma2,
        seed=seed,
        max_workers=max_workers
    )
