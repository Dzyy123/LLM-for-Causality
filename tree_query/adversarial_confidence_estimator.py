"""
Adversarial Confidence Estimator Module

Implements confidence estimation using adversarial arguments.
Includes FULL original explanation in adversarially-influenced sampling prompts.
Uses multi-threading for parallel LLM requests and modular design.
"""

import random
import logging
from typing import List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from llm_utils import LocalLLMClient, OnlineLLMClient
try:
    from llm_utils.report_logger import get_report_logger
except ImportError:
    get_report_logger = None

from tree_query.utils import extract_yes_no_from_response
from tree_query.config_loader import get_config


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
    """Container for a single adversarial argument."""
    dtype: str
    prompt: str
    argument: str


@dataclass
class AdversarySet:
    """Container for a complete set of adversarial arguments."""
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
                                original_label: int, original_answer_text: str,
                                adversarial_argument: str) -> str:
        """Build prompt for sampling with adversarial argument (includes full original answer)."""
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
            adversarial_argument=adversarial_argument,
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
                         original_answers: List[OriginalAnswer]) -> List[AdversarySet]:
        """Generate all adversarial sets (m sets, one per original answer) in parallel."""
        total_sets = len(original_answers)
        logger.info(f"Generating {total_sets} adversarial sets")
        
        # Build task list
        tasks = []
        set_index = 0
        for orig_answer in original_answers:
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
    """Samples answers after presenting adversarial arguments (with full original answer)."""
    
    def __init__(self, executor: LLMRequestExecutor, prompt_builder: PromptBuilder, max_workers: int = 15):
        self.executor = executor
        self.prompt_builder = prompt_builder
        self.max_workers = max_workers
    
    def sample_with_adversarial(self, question: str, adversarial_set: AdversarySet,
                               dtype: str) -> AdversarySample:
        """Sample with one adversarial type (includes full original answer text)."""
        adversarial = getattr(adversarial_set, dtype)
        
        # Build sampling prompt: pass the full original answer text
        sampling_prompt = self.prompt_builder.build_sampling_prompt(
            dtype, question, adversarial_set.original_label, 
            adversarial_set.original_answer, adversarial.argument
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
        """Sample with all adversarial arguments (m × 3 samples) in parallel."""
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
    
    def __init__(self, gamma1: float = 1e-6, gamma2: float = 1e-6,
                 weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weights = weights
    
    def compute_majority_aligned_rate(self, majority_label: int, 
                                      adversarial_results: List[int]) -> float:
        """Compute proportion of results matching the majority label."""
        if not adversarial_results:
            return 0.0
        aligned = sum(1 for r in adversarial_results if r == majority_label)
        return aligned / len(adversarial_results)
    
    def compute_label_distribution(self, adversarial_results: List[int]) -> Dict[str, int]:
        """Compute Yes/No distribution."""
        yes_count = sum(1 for r in adversarial_results if r == 1)
        no_count = len(adversarial_results) - yes_count
        return {'yes': yes_count, 'no': no_count}
    
    def compute_confidence_score(self, p0: float, p1: float, 
                                 p2: float, p3: float, p0_raw: float) -> float:
        """Compute final confidence score using weighted deviation formula."""
        if p0_raw == 0:
            return 0.0
        
        p_values = [p1, p2, p3]
        weighted_deviation_sum = 0.0
        for i, p_i in enumerate(p_values):
            weighted_deviation_sum += self.weights[i] * abs(p_i - p0_raw) / p0_raw
        
        confidence = p0 * (1 - weighted_deviation_sum)
        
        return confidence
    
    def calculate_all_metrics(self, expert_result: int,
                             adversarial_samples: List[AdversarySample],
                             original_answers: List[OriginalAnswer]) -> Dict[str, Any]:
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
        
        # Determine majority label from original answers
        original_labels = [ans.label for ans in original_answers]
        yes_count = sum(1 for label in original_labels if label == 1)
        no_count = len(original_labels) - yes_count
        
        # Majority label y*
        majority_label = 1 if yes_count >= no_count else 0
        
        # p0_raw is the proportion of the majority label (range: [0.5, 1])
        p0_raw = max(yes_count, no_count) / len(original_labels) if original_labels else 1.0
        
        # Apply adjustment: p0 = p0_raw * 2 - 1 to map [0.5, 1] -> [0, 1]
        p0 = p0_raw * 2 - 1
        
        # Calculate majority-aligned rates for each adversarial type
        # p_j = proportion of adversarially-influenced conclusions that equal y*
        majority_aligned_rates = {}
        label_distributions = {}
        
        for dtype in ['contrarian', 'deceiver', 'hater']:
            type_samples = [s for s in adversarial_samples if s.adversarial_type == dtype]
            
            # Calculate proportion of results matching the majority label
            majority_aligned_rates[dtype] = self.compute_majority_aligned_rate(
                majority_label, [s.label for s in type_samples]
            )
            
            # Label distribution
            label_distributions[dtype] = self.compute_label_distribution([s.label for s in type_samples])
        
        # p_j values as defined in the paper
        p1 = majority_aligned_rates['contrarian']
        p2 = majority_aligned_rates['deceiver']
        p3 = majority_aligned_rates['hater']
        
        # Calculate confidence score with p0_raw as denominator
        confidence_score = self.compute_confidence_score(p0, p1, p2, p3, p0_raw)
        
        # Calculate backward compatibility metrics
        consistency_rates = {}
        flip_rates = {}
        for dtype in ['contrarian', 'deceiver', 'hater']:
            type_samples = [s for s in adversarial_samples if s.adversarial_type == dtype]
            # Consistency rate: proportion matching their own original labels
            total_consistent = sum(1 for s in type_samples if s.label == s.original_label)
            consistency_rates[dtype] = total_consistent / len(type_samples) if type_samples else 0.0
            flip_rates[dtype] = 1 - consistency_rates[dtype]
        
        avg_flip_rate = sum(flip_rates.values()) / 3
        robustness_score = 1 - avg_flip_rate
        
        return {
            'majority_label': majority_label,
            'majority_aligned_rates': majority_aligned_rates,
            'consistency_rates': consistency_rates,
            'flip_rates': flip_rates,
            'label_distributions': label_distributions,
            'detailed_tracking': detailed_tracking,
            'avg_flip_rate': avg_flip_rate,
            'robustness_score': robustness_score,
            'p0_raw': p0_raw,
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'confidence_score': confidence_score
        }


class AdversarialConfidenceEstimator:
    """
    Main estimator class coordinating all components.
    Uses multi-threading for parallel LLM requests.
    Includes full original explanation in adversarially-influenced sampling.
    """
    
    def __init__(self, client: Union[LocalLLMClient, OnlineLLMClient],
                 m_samples: int = 20,
                 gamma1: float = 1e-6, gamma2: float = 1e-6,
                 seed: int = None, max_workers: int = 10,
                 weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)):
        """
        Initialize the estimator.
        
        Args:
            client: LLM client for requests
            m_samples: Number of original answer samples
            gamma1: Stabilization parameter for numerator
            gamma2: Stabilization parameter for denominator
            seed: Random seed (auto-increments per request)
            max_workers: Maximum number of threads for parallel LLM requests
            weights: Tuple of 3 floats (lambda_1, lambda_2, lambda_3) summing to 1,
                     representing weights for contrarian, deceiver, hater adversarial arguments
        """
        # Validate weights
        if len(weights) != 3:
            raise ValueError("weights must be a tuple of 3 floats")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {sum(weights)}")
        
        self.m_samples = m_samples
        self.initial_seed = seed
        self.max_workers = max_workers
        self.weights = weights
        
        # Initialize components
        self.seed_manager = SeedManager(seed)
        self.prompt_builder = PromptBuilder()
        self.executor = LLMRequestExecutor(client, self.seed_manager, max_workers)
        self.adversarial_generator = AdversaryGenerator(self.executor, self.prompt_builder, max_workers)
        self.adversarial_sampler = AdversarySampler(self.executor, self.prompt_builder, max_workers)
        self.metrics_calculator = MetricsCalculator(gamma1, gamma2, weights)
    
    def estimate_confidence(self, expert, result: int) -> Dict[str, Any]:
        """
        Estimate confidence score with adversary-based testing.
        
        Args:
            expert: Expert instance with base_prompt
            result: Expert's answer (0 or 1)
        
        Returns:
            Dictionary with all metrics and intermediate data
        """
        question = expert.base_prompt
        
        logger.info("Starting confidence estimation")
        logger.info(f"Expert: {expert.expert_type}, Result: {'Yes' if result == 1 else 'No'}")
        logger.info(f"Config: m={self.m_samples}, total={self.m_samples * 3}")
        
        # Add subsection marker for original sampling phase
        if get_report_logger is not None:
            report_logger = get_report_logger()
            if report_logger.enabled and report_logger.report_file is not None:
                with report_logger._lock:
                    with open(report_logger.report_file, 'a', encoding='utf-8') as f:
                        f.write(f"#### Phase 1: Original Answer Sampling (m={self.m_samples})\n\n")
                        f.write(f"Generating {self.m_samples} independent samples with the same question...\n\n")
        
        # Step 1: Sample original answers
        original_answers = sample_original_answers(self.executor, question, self.m_samples)
        
        # Add subsection marker for adversarial generation phase
        if get_report_logger is not None:
            report_logger = get_report_logger()
            if report_logger.enabled and report_logger.report_file is not None:
                with report_logger._lock:
                    with open(report_logger.report_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n#### Phase 2: Adversarial Argument Generation (m={self.m_samples})\n\n")
                        f.write(f"For each of the {self.m_samples} original answers, generating adversarial arguments (contrarian, deceiver, hater)...\n\n")
        
        # Step 2: Generate adversarial sets
        adversarial_sets = self.adversarial_generator.generate_all_sets(
            question, original_answers
        )
        
        # Add subsection marker for adversarial sampling phase
        if get_report_logger is not None:
            report_logger = get_report_logger()
            if report_logger.enabled and report_logger.report_file is not None:
                with report_logger._lock:
                    with open(report_logger.report_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n#### Phase 3: Adversarially-Influenced Sampling (m×3={self.m_samples * 3})\n\n")
                        f.write(f"For each adversarial argument set, generating responses influenced by contrarian, deceiver, and hater arguments...\n\n")
        
        # Step 3: Sample with adversarial arguments
        adversarial_samples = self.adversarial_sampler.sample_all(
            question, adversarial_sets
        )
        
        # Add subsection marker for metrics calculation
        if get_report_logger is not None:
            report_logger = get_report_logger()
            if report_logger.enabled and report_logger.report_file is not None:
                with report_logger._lock:
                    with open(report_logger.report_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n#### Phase 4: Confidence Metrics Calculation\n\n")
                        f.write(f"Analyzing flip rates and computing final confidence score...\n\n")
        
        # Step 4: Calculate metrics
        logger.info("Calculating metrics")
        metrics = self.metrics_calculator.calculate_all_metrics(
            result, adversarial_samples, original_answers
        )
        
        logger.info(f"Confidence: {metrics['confidence_score']:.4f}, Robustness: {metrics['robustness_score']:.2%}")
        
        # Build comprehensive result
        return {
            **metrics,
            'result': result,
            'expert_type': expert.expert_type,
            'm_samples': self.m_samples,
            'total_samples': self.m_samples * 3,
            'original_answers': original_answers,
            'adversarial_sets': adversarial_sets,
            'adversarial_samples': adversarial_samples
        }


def create_adversarial_confidence_estimator(
    client: Union[LocalLLMClient, OnlineLLMClient],
    m_samples: int = 50,
    gamma1: float = 1e-6, gamma2: float = 1e-6,
    seed: int = None, max_workers: int = 10,
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)
) -> AdversarialConfidenceEstimator:
    """Factory function to create estimator."""
    return AdversarialConfidenceEstimator(
        client=client,
        m_samples=m_samples,
        gamma1=gamma1,
        gamma2=gamma2,
        seed=seed,
        max_workers=max_workers,
        weights=weights
    )
