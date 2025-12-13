"""
Distractor Confidence Estimator Module

Implements confidence estimation using adversarial distractor arguments.
Includes FULL original explanation in distracted sampling prompts.
Uses multi-threading for parallel LLM requests and modular design.
"""

import math
import random
import logging
from typing import List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from llm_utils import LocalLLMClient, OnlineLLMClient
from utils import extract_yes_no_from_response
from config_loader import get_config


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class OriginalAnswer:
    """Container for original answer data."""
    text: str
    label: int
    index: int


@dataclass
class DistractorArgument:
    """Container for a single distractor argument."""
    dtype: str
    prompt: str
    argument: str


@dataclass
class DistractorSet:
    """Container for a complete set of distractor arguments."""
    set_index: int
    original_answer: str
    original_label: int
    contrarian: DistractorArgument
    deceiver: DistractorArgument
    hater: DistractorArgument


@dataclass
class DistractedSample:
    """Container for a distracted sample result."""
    set_index: int
    distractor_type: str
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
    """Builds prompts for distractor generation and sampling."""
    
    def __init__(self):
        config = get_config()
        self.contrarian_template = config.get_distractor_prompt('contrarian')
        self.deceiver_template = config.get_distractor_prompt('deceiver')
        self.hater_template = config.get_distractor_prompt('hater')
        self.single_contrarian_template = config.get_distractor_prompt('single_contrarian_template')
        self.single_deceiver_template = config.get_distractor_prompt('single_deceiver_template')
        self.single_hater_template = config.get_distractor_prompt('single_hater_template')
    
    def build_generation_prompt(self, dtype: str, question: str, 
                                answer: str, conclusion: str, 
                                opposite_conclusion: str) -> str:
        """Build prompt for generating distractor argument."""
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
                                original_label: int, original_answer_text: str,
                                distractor_argument: str) -> str:
        """Build prompt for sampling with distractor (includes full original answer)."""
        template_map = {
            'contrarian': self.single_contrarian_template,
            'deceiver': self.single_deceiver_template,
            'hater': self.single_hater_template
        }
        template = template_map[dtype]
        original_conclusion = "Yes" if original_label == 1 else "No"
        
        # Include full original answer text in the format string
        # The template expects {original_conclusion}, we'll provide more context
        return template.format(
            question=question,
            separator="",
            distractor_argument=distractor_argument,
            original_conclusion=f"{original_conclusion}\n\nThe original full response was:\n{original_answer_text}"
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


class DistractorGenerator:
    """Generates distractor arguments."""
    
    def __init__(self, executor: LLMRequestExecutor, prompt_builder: PromptBuilder, max_workers: int = 10):
        self.executor = executor
        self.prompt_builder = prompt_builder
        self.max_workers = max_workers
    
    def generate_single_set(self, question: str, original_answer: str, 
                           original_label: int) -> Tuple[DistractorArgument, DistractorArgument, DistractorArgument]:
        """Generate one complete set of three distractors in parallel."""
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
        
        # Build DistractorArgument objects
        distractor_map = {}
        for dtype, argument in results:
            # Find the prompt used
            prompt = next(p for dt, p in prompts if dt == dtype)
            distractor_map[dtype] = DistractorArgument(
                dtype=dtype,
                prompt=prompt,
                argument=argument
            )
        
        return (distractor_map['contrarian'], 
                distractor_map['deceiver'], 
                distractor_map['hater'])
    
    def generate_all_sets(self, question: str, 
                         original_answers: List[OriginalAnswer],
                         k2_samples: int) -> List[DistractorSet]:
        """Generate all distractor sets (k1 × k2 sets) in parallel."""
        total_sets = len(original_answers) * k2_samples
        logger.info(f"Generating {total_sets} distractor sets")
        
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
        distractor_sets = []
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
                
                distractor_sets.append(DistractorSet(
                    set_index=set_idx,
                    original_answer=orig_text,
                    original_label=orig_label,
                    contrarian=contrarian,
                    deceiver=deceiver,
                    hater=hater
                ))
        
        # Sort by set_index to maintain order
        distractor_sets.sort(key=lambda x: x.set_index)
        logger.info(f"Completed {len(distractor_sets)} distractor sets")
        return distractor_sets


class DistractedSampler:
    """Samples answers after presenting distractors (with full original answer)."""
    
    def __init__(self, executor: LLMRequestExecutor, prompt_builder: PromptBuilder, max_workers: int = 15):
        self.executor = executor
        self.prompt_builder = prompt_builder
        self.max_workers = max_workers
    
    def sample_with_distractor(self, question: str, distractor_set: DistractorSet,
                               dtype: str) -> DistractedSample:
        """Sample with one distractor type (includes full original answer text)."""
        distractor = getattr(distractor_set, dtype)
        
        # Build sampling prompt: pass the full original answer text
        sampling_prompt = self.prompt_builder.build_sampling_prompt(
            dtype, question, distractor_set.original_label, 
            distractor_set.original_answer, distractor.argument
        )
        
        # Execute request
        response = self.executor.execute_single(sampling_prompt)
        label = extract_yes_no_from_response(response)
        
        return DistractedSample(
            set_index=distractor_set.set_index,
            distractor_type=dtype,
            original_label=distractor_set.original_label,
            sampling_prompt=sampling_prompt,
            response=response,
            label=label
        )
    
    def sample_all(self, question: str, 
                   distractor_sets: List[DistractorSet]) -> List[DistractedSample]:
        """Sample with all distractors (k1 × k2 × 3 samples) in parallel."""
        total_samples = len(distractor_sets) * 3
        logger.info(f"Sampling {total_samples} distracted responses")
        
        # Build task list
        tasks = []
        for dset in distractor_sets:
            for dtype in ['contrarian', 'deceiver', 'hater']:
                tasks.append((question, dset, dtype))
        
        # Execute in parallel
        samples = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.sample_with_distractor,
                    task[0], task[1], task[2]
                ): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                sample = future.result()
                samples.append(sample)
        
        # Sort by set_index and distractor_type
        samples.sort(key=lambda x: (x.set_index, x.distractor_type))
        logger.info(f"Completed {len(samples)} distracted responses")
        return samples


class MetricsCalculator:
    """Calculates confidence metrics."""
    
    def __init__(self, gamma1: float = 1e-6, gamma2: float = 1e-6,
                 weights: Tuple[float, float, float] = (1/4, 1/4, 1/2)):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weights = weights
    
    def compute_flip_rate(self, original_result: int, 
                         distracted_results: List[int]) -> float:
        """Compute flip rate for one distractor type."""
        if not distracted_results:
            return 0.0
        flips = sum(1 for r in distracted_results if r != original_result)
        return flips / len(distracted_results)
    
    def compute_label_distribution(self, distracted_results: List[int]) -> Dict[str, int]:
        """Compute Yes/No distribution."""
        yes_count = sum(1 for r in distracted_results if r == 1)
        no_count = len(distracted_results) - yes_count
        return {'yes': yes_count, 'no': no_count}
    
    def compute_confidence_score(self, p0: float, p1: float, 
                                 p2: float, p3: float) -> float:
        """Compute final confidence score using weighted deviation formula."""
        if p0 == 0:
            return 0.0
        
        p_values = [p1, p2, p3]
        weighted_deviation_sum = 0.0
        for i, p_i in enumerate(p_values):
            weighted_deviation_sum += self.weights[i] * abs(p_i - p0) / p0
        
        confidence = p0 * (1 - weighted_deviation_sum)
        
        return max(0.0, confidence)
    
    def calculate_all_metrics(self, expert_result: int,
                             distracted_samples: List[DistractedSample]) -> Dict[str, Any]:
        """Calculate all confidence metrics."""
        # Build detailed tracking: group by set_index to show each original answer's journey
        detailed_tracking = {}
        for sample in distracted_samples:
            if sample.set_index not in detailed_tracking:
                detailed_tracking[sample.set_index] = {
                    'original_label': sample.original_label,
                    'contrarian': None,
                    'deceiver': None,
                    'hater': None
                }
            detailed_tracking[sample.set_index][sample.distractor_type] = sample.label
        
        # Calculate flip rates and label distributions based on EACH distractor set's original label
        # Group samples by distractor type for proper flip rate calculation
        flip_rates = {}
        label_distributions = {}
        
        for dtype in ['contrarian', 'deceiver', 'hater']:
            type_samples = [s for s in distracted_samples if s.distractor_type == dtype]
            
            # Calculate flip rate: compare each distracted result with its OWN original label
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


class DistractorConfidenceEstimator:
    """
    Main estimator class coordinating all components.
    Uses multi-threading for parallel LLM requests.
    Includes full original explanation in distracted sampling.
    """
    
    def __init__(self, client: Union[LocalLLMClient, OnlineLLMClient],
                 k1_samples: int = 20, k2_samples: int = 1,
                 gamma1: float = 1e-6, gamma2: float = 1e-6,
                 seed: int = None, max_workers: int = 10,
                 weights: Tuple[float, float, float] = (1/4, 1/4, 1/2)):
        """
        Initialize the estimator.
        
        Args:
            client: LLM client for requests
            k1_samples: Number of original answer samples
            k2_samples: Number of distractor sets per original answer
            gamma1: Stabilization parameter for numerator
            gamma2: Stabilization parameter for denominator
            seed: Random seed (auto-increments per request)
            max_workers: Maximum number of threads for parallel LLM requests
            weights: Tuple of 3 floats (lambda_1, lambda_2, lambda_3) summing to 1,
                     representing weights for contrarian, deceiver, hater distractors
        """
        # Validate weights
        if len(weights) != 3:
            raise ValueError("weights must be a tuple of 3 floats")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {sum(weights)}")
        
        self.k1_samples = k1_samples
        self.k2_samples = k2_samples
        self.initial_seed = seed
        self.max_workers = max_workers
        self.weights = weights
        
        # Initialize components
        self.seed_manager = SeedManager(seed)
        self.prompt_builder = PromptBuilder()
        self.executor = LLMRequestExecutor(client, self.seed_manager, max_workers)
        self.distractor_generator = DistractorGenerator(self.executor, self.prompt_builder, max_workers)
        self.distracted_sampler = DistractedSampler(self.executor, self.prompt_builder, max_workers)
        self.metrics_calculator = MetricsCalculator(gamma1, gamma2, weights)
    
    def estimate_confidence(self, expert, result: int) -> Dict[str, Any]:
        """
        Estimate confidence score with distractor-based testing.
        
        Args:
            expert: Expert instance with base_prompt
            result: Expert's answer (0 or 1)
        
        Returns:
            Dictionary with all metrics and intermediate data
        """
        question = expert.base_prompt
        
        logger.info("Starting confidence estimation")
        logger.info(f"Expert: {expert.expert_type}, Result: {'Yes' if result == 1 else 'No'}")
        logger.info(f"Config: k1={self.k1_samples}, k2={self.k2_samples}, total={self.k1_samples * self.k2_samples * 3}")
        
        # Step 1: Sample original answers
        original_answers = sample_original_answers(self.executor, question, self.k1_samples)
        
        # Step 2: Generate distractor sets
        distractor_sets = self.distractor_generator.generate_all_sets(
            question, original_answers, self.k2_samples
        )
        
        # Step 3: Sample with distractors
        distracted_samples = self.distracted_sampler.sample_all(
            question, distractor_sets
        )
        
        # Step 4: Calculate metrics
        logger.info("Calculating metrics")
        metrics = self.metrics_calculator.calculate_all_metrics(
            result, distracted_samples
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
            'distractor_sets': distractor_sets,
            'distracted_samples': distracted_samples
        }


def create_distractor_confidence_estimator(
    client: Union[LocalLLMClient, OnlineLLMClient],
    k1_samples: int = 50, k2_samples: int = 1,
    gamma1: float = 1e-6, gamma2: float = 1e-6,
    seed: int = None, max_workers: int = 10,
    weights: Tuple[float, float, float] = (1/4, 1/4, 1/2)
) -> DistractorConfidenceEstimator:
    """Factory function to create estimator."""
    return DistractorConfidenceEstimator(
        client=client,
        k1_samples=k1_samples,
        k2_samples=k2_samples,
        gamma1=gamma1,
        gamma2=gamma2,
        seed=seed,
        max_workers=max_workers,
        weights=weights
    )
