"""
Causal Discovery Framework with Adversarial-Based Confidence Estimation

A comprehensive framework for causal discovery that integrates:
- Tree-based causal inference logic
- MoE (Mixture of Experts) architecture
- Adversary-based confidence estimation
- Multi-calibration for probability adjustment
- Prior and posterior causal graph generation

This replaces the method-based confidence (frequency/probability/logit) with
the more robust AdversarialConfidenceEstimator.
"""

import logging
import json
import hashlib
from typing import List, Dict, Any, Tuple, Union, Optional
from itertools import combinations
from pathlib import Path

from llm_utils import LocalLLMClient, OnlineLLMClient
try:
    from llm_utils.report_logger import get_report_logger
except ImportError:
    get_report_logger = None

from tree_query.experts import (
    BackdoorPathExpert, 
    IndependenceExpert, 
    LatentConfounderExpert, 
    CausalDirectionExpert
)
from tree_query.adversarial_confidence_estimator import create_adversarial_confidence_estimator
from tree_query.expert_router import ExpertRouter
from tree_query.utils import aggregate_expert_results
from tree_query.config_loader import get_config


# Configure module logger
logger = logging.getLogger(__name__)


class CausalDiscoveryFramework:
    """
    Main framework orchestrating causal discovery with adversary-based confidence.
    
    Architecture:
    1. Tree-based query logic for systematic causal relationship determination
    2. MoE expert routing for diverse perspectives
    3. Adversary-based confidence estimation for robustness
    4. Multi-calibration for probability refinement
    """
    
    def __init__(
        self,
        client: Union[LocalLLMClient, OnlineLLMClient],
        all_variables: List[str],
        m_samples: int = 5,
        seed: int = 42,
        max_workers: int = 10,
        trust_confidence: float = 0.75
    ):
        """
        Initialize the causal discovery framework.
        
        Args:
            client: LLM client for experts and confidence estimation
            all_variables: List of all variables in the causal system
            m_samples: Number of original answer samples for confidence estimation
            seed: Random seed for reproducibility
            max_workers: Maximum number of threads for parallel LLM requests
            trust_confidence: Confidence threshold (0-1) to trust a result without
                exploring other branches. Default is 0.75.
        """
        if not 0 <= trust_confidence <= 1:
            raise ValueError("trust_confidence must be between 0 and 1")
        
        self.client = client
        self.all_variables = all_variables
        self.router = ExpertRouter(client)
        self.config = get_config()
        self.trust_confidence = trust_confidence
        self.seed = seed
        
        # Pre-compute expert selections for all variable pairs
        # Try to load from cache first, otherwise compute and save
        self._expert_cache = self._load_or_initialize_expert_cache()
        
        # Initialize adversary-based confidence estimator
        self.confidence_estimator = create_adversarial_confidence_estimator(
            client=client,
            m_samples=m_samples,
            seed=seed,
            max_workers=max_workers
        )
        
        logger.info(f"Initialized CausalDiscoveryFramework with {len(all_variables)} variables")
        logger.info(f"Confidence estimator: m={m_samples}")
        logger.info(f"Trust confidence threshold: {trust_confidence}")
        logger.info(f"Pre-computed expert selections for {len(self._expert_cache)} variable-question pairs")
    
    def _get_cache_filepath(self) -> Path:
        """
        Get the filepath for the expert cache based on variable names.
        
        Returns:
            Path to the cache file
        """
        # Create a hash of sorted variable names for unique identification
        var_str = "_".join(sorted(self.all_variables))
        var_hash = hashlib.md5(var_str.encode()).hexdigest()[:8]
        
        # Create cache directory
        cache_dir = Path(".expert_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Cache filename includes hash and variable count
        cache_file = cache_dir / f"expert_cache_{len(self.all_variables)}vars_{var_hash}.json"
        return cache_file
    
    def _save_expert_cache(self, cache: Dict[Tuple[str, str, str], List[str]]) -> None:
        """
        Save expert cache to disk.
        
        Args:
            cache: Expert cache dictionary to save
        """
        cache_file = self._get_cache_filepath()
        
        try:
            # Convert tuple keys to strings for JSON serialization
            serializable_cache = {
                f"{x1}||{x2}||{qtype}": experts
                for (x1, x2, qtype), experts in cache.items()
            }
            
            cache_data = {
                "variables": sorted(self.all_variables),
                "cache": serializable_cache,
                "version": "1.0"
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Expert cache saved to: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save expert cache: {e}")
    
    def _load_expert_cache(self) -> Optional[Dict[Tuple[str, str, str], List[str]]]:
        """
        Load expert cache from disk if it exists.
        
        Returns:
            Loaded cache dictionary or None if cache doesn't exist or is invalid
        """
        cache_file = self._get_cache_filepath()
        
        if not cache_file.exists():
            logger.info(f"No expert cache found at: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Verify variables match
            cached_vars = set(cache_data.get("variables", []))
            current_vars = set(self.all_variables)
            
            if cached_vars != current_vars:
                logger.warning(f"Cache variables don't match current variables, rebuilding cache")
                return None
            
            # Convert string keys back to tuples
            serializable_cache = cache_data.get("cache", {})
            expert_cache = {}
            
            for key_str, experts in serializable_cache.items():
                parts = key_str.split("||")
                if len(parts) == 3:
                    x1, x2, qtype = parts
                    expert_cache[(x1, x2, qtype)] = experts
            
            logger.info(f"Loaded expert cache from: {cache_file} ({len(expert_cache)} entries)")
            return expert_cache
            
        except Exception as e:
            logger.warning(f"Failed to load expert cache: {e}")
            return None
    
    def _load_or_initialize_expert_cache(self) -> Dict[Tuple[str, str, str], List[str]]:
        """
        Load expert cache from disk if available, otherwise initialize and save.
        
        Returns:
            Dict mapping (x1, x2, question_type) to list of selected expert types
        """
        # Try to load existing cache
        cached_data = self._load_expert_cache()
        if cached_data is not None:
            return cached_data
        
        # Cache not found or invalid, initialize new cache
        logger.info("Initializing expert cache (this may take a while)...")
        expert_cache = self._initialize_expert_cache()
        
        # Save the newly created cache
        self._save_expert_cache(expert_cache)
        
        return expert_cache
    
    def _initialize_expert_cache(self) -> Dict[Tuple[str, str, str], List[str]]:
        """
        Pre-compute expert selections for all variable pairs and question types.
        This will call the LLM router for each combination.
        
        Returns:
            Dict mapping (x1, x2, question_type) to list of selected expert types
        """
        logger.info("Computing expert selections (calling LLM router for each pair)...")
        expert_cache = {}
        question_types = ['backdoor_path', 'independence', 'latent_confounder', 'causal_direction']
        
        total_combinations = len(self.all_variables) * (len(self.all_variables) - 1) * len(question_types)
        logger.info(f"Need to compute {total_combinations} expert selections...")
        
        processed = 0
        # Generate all variable pairs
        for i, x1 in enumerate(self.all_variables):
            for x2 in self.all_variables:
                if x1 != x2:  # Skip self-pairs
                    for question_type in question_types:
                        processed += 1
                        if processed % 10 == 0:
                            logger.info(f"  Progress: {processed}/{total_combinations}")
                        
                        cache_key = (x1, x2, question_type)
                        selected_experts = self.router.select_experts(question_type, x1, x2, top_k=3)
                        expert_cache[cache_key] = selected_experts
                        
        logger.info(f"Expert cache initialized with {len(expert_cache)} entries")
        return expert_cache
    
    def _aggregate_experts_with_confidence(
        self, 
        expert_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate expert results with equal voting power.
        
        Strategy:
        1. Each expert has EQUAL voting power (1 vote)
        2. Count votes for each possible label
        3. Select label with most votes (simple majority)
        4. Final confidence is the average confidence of experts supporting the winning label
        
        Args:
            expert_results: List of dicts with 'expert', 'label', 'confidence'
            
        Returns:
            Dict with aggregated 'label', 'confidence', 'method', and 'details'
        """
        from collections import defaultdict
        import numpy as np
        
        # Equal voting: count votes for each label
        label_votes = defaultdict(lambda: {"vote_count": 0, "confidences": [], "experts": []})
        
        for result in expert_results:
            label = result["label"]
            confidence = result["confidence"]
            expert = result["expert"]
            
            label_votes[label]["vote_count"] += 1  # Each expert gets 1 vote
            label_votes[label]["confidences"].append(confidence)
            label_votes[label]["experts"].append(expert)
        
        # Select label with highest vote count
        winning_label = max(label_votes.items(), key=lambda x: x[1]["vote_count"])[0]
        
        # Calculate final confidence as average of supporting experts
        supporting_confidences = label_votes[winning_label]["confidences"]
        final_confidence = np.mean(supporting_confidences)
        
        aggregation_details = {
            "label_vote_counts": {k: v["vote_count"] for k, v in label_votes.items()},
            "winning_experts": label_votes[winning_label]["experts"],
            "winning_expert_count": len(label_votes[winning_label]["experts"]),
            "total_expert_count": len(expert_results)
        }
        
        return {
            "label": winning_label,
            "confidence": final_confidence,
            "method": "equal_voting",
            "details": aggregation_details
        }
    
    def _get_cached_experts(self, question_type: str, x1: str, x2: str) -> List[str]:
        """
        Get pre-computed expert selection from cache.
        
        Args:
            question_type: Type of question
            x1: First variable
            x2: Second variable
            
        Returns:
            List of selected expert type identifiers
        """
        cache_key = (x1, x2, question_type)
        return self._expert_cache.get(cache_key, [])
    
    def _run_expert_with_confidence(
        self,
        expert_class: type,
        question_type: str,
        x1: str,
        x2: str
    ) -> Dict[str, Any]:
        """
        Run expert judgment with individual adversary-based confidence estimation.
        
        Each expert answer goes through its own adversarial validation.
        No immediate aggregation - all expert results with their individual
        confidence scores are returned for final aggregation.
        
        Args:
            expert_class: The expert class to use
            question_type: Type of question for routing
            x1: First variable
            x2: Second variable
            
        Returns:
            Dict with individual expert results and final aggregated decision
        """
        # Add section to report
        if get_report_logger is not None:
            report_logger = get_report_logger()
            if report_logger.enabled and report_logger.report_file is not None:
                report_logger.add_section(
                    f"Query: {question_type.replace('_', ' ').title()} ({x1}, {x2})",
                    f"Checking relationship between **{x1}** and **{x2}** using question type: `{question_type}`\n\n"
                    f"This query will use multiple experts with individual adversarial validation."
                )
        
        # Get pre-computed expert selection from cache
        selected_expert_types = self._get_cached_experts(question_type, x1, x2)
        logger.info(f"Using cached experts for '{question_type}': {selected_expert_types}")
        
        # Execute expert judgments with individual adversarial validation
        expert_results_with_confidence = []
        for expert_type in selected_expert_types:
            try:
                # Create expert instance
                expert = expert_class(
                    base_prompt="",  # Will be generated
                    x1=x1,
                    x2=x2,
                    client=self.client,
                    all_variables=self.all_variables,
                    expert_type=expert_type,
                    seed=self.seed
                )
                expert.base_prompt = expert.generate_question()
                
                # Get the full prompt that will be sent to LLM
                full_prompt = expert.create_expert_prompt()
                
                # Get expert judgment
                result = expert.judge()
                initial_label = result["label"]
                
                logger.info(f"Expert {expert_type}: initial label={initial_label}")
                
                # Add expert section to report
                if get_report_logger is not None:
                    report_logger = get_report_logger()
                    if report_logger.enabled and report_logger.report_file is not None:
                        expert_label_text = "Yes (1)" if initial_label == 1 else "No (0)"
                        report_logger.add_section(
                            f"Expert: {expert_type}",
                            f"**Expert Type:** {expert_type}\n\n"
                            f"**Initial Judgment:** {expert_label_text}\n\n"
                            f"**Full Prompt:**\n```\n{full_prompt}\n```\n\n"
                            f"**Response:**\n```\n{result.get('response', '')}\n```\n\n"
                            f"---\n\n"
                            f"**Starting Adversarial Confidence Estimation...**\n\n"
                            f"Configuration: m_samples={self.confidence_estimator.m_samples}\n\n"
                            f"This will generate {self.confidence_estimator.m_samples} original samples and "
                            f"{self.confidence_estimator.m_samples * 3} "
                            f"adversarially-influenced samples.\n"
                        )
                
                # Individual adversarial confidence estimation for this expert
                try:
                    confidence_result = self.confidence_estimator.estimate_confidence(
                        expert,
                        initial_label
                    )
                    confidence_score = confidence_result["confidence_score"]
                    logger.info(f"  -> Adversarial confidence: {confidence_score:.4f}")
                    
                    # Add confidence result to report
                    if get_report_logger is not None:
                        report_logger = get_report_logger()
                        if report_logger.enabled and report_logger.report_file is not None:
                            flip_rates = confidence_result.get('flip_rates', {})
                            
                            # Build detailed Yes/No results table
                            original_answers = confidence_result.get('original_answers', [])
                            adversarial_samples = confidence_result.get('adversarial_samples', [])
                            detailed_tracking = confidence_result.get('detailed_tracking', {})
                            
                            # Original samples summary
                            original_summary = "\n**Original Samples (m={}) for Expert `{}`:**\n\n".format(
                                len(original_answers), expert_type
                            )
                            original_summary += "| Sample | Expert | Label | Result |\n|--------|--------|-------|--------|\n"
                            for i, ans in enumerate(original_answers, 1):
                                label_text = "Yes" if ans.label == 1 else "No"
                                original_summary += f"| {i} | {expert_type} | {ans.label} | {label_text} |\n"
                            
                            # Calculate label distribution in original samples
                            yes_count = sum(1 for ans in original_answers if ans.label == 1)
                            no_count = len(original_answers) - yes_count
                            original_summary += f"\n**Original Label Distribution:** Yes={yes_count}, No={no_count}\n\n"
                            
                            # Adversarial samples summary by set
                            adversarial_summary = "\n**Adversarially-Influenced Samples (m×3={}) for Expert `{}`:**\n\n".format(
                                len(adversarial_samples), expert_type
                            )
                            adversarial_summary += "| Set | Expert | Original | Contrarian | Deceiver | Hater |\n"
                            adversarial_summary += "|-----|--------|----------|------------|----------|-------|\n"
                            
                            for set_idx in sorted(detailed_tracking.keys()):
                                tracking = detailed_tracking[set_idx]
                                orig = "Yes" if tracking['original_label'] == 1 else "No"
                                contr = "Yes" if tracking.get('contrarian') == 1 else "No"
                                decv = "Yes" if tracking.get('deceiver') == 1 else "No"
                                hatr = "Yes" if tracking.get('hater') == 1 else "No"
                                
                                # Mark flips with emoji
                                if tracking['original_label'] != tracking.get('contrarian'):
                                    contr += " ⚠️"
                                if tracking['original_label'] != tracking.get('deceiver'):
                                    decv += " ⚠️"
                                if tracking['original_label'] != tracking.get('hater'):
                                    hatr += " ⚠️"
                                
                                adversarial_summary += f"| {set_idx} | {expert_type} | {orig} | {contr} | {decv} | {hatr} |\n"
                            
                            adversarial_summary += "\n*⚠️ = Label flipped from original*\n\n"
                            
                            # Label distribution analysis
                            label_dist = confidence_result.get('label_distributions', {})
                            dist_summary = "\n**Adversarial Label Distributions:**\n\n"
                            for adv_type in ['contrarian', 'deceiver', 'hater']:
                                dist = label_dist.get(adv_type, {})
                                dist_summary += f"- **{adv_type.title()}:** Yes={dist.get('yes', 0)}, No={dist.get('no', 0)}\n"
                            
                            report_logger.add_section(
                                f"Confidence Result for {expert_type}",
                                f"**Final Confidence Score:** {confidence_score:.4f}\n\n"
                                f"**Robustness Metrics:**\n"
                                f"- Contrarian flip rate: {flip_rates.get('contrarian', 0):.2%}\n"
                                f"- Deceiver flip rate: {flip_rates.get('deceiver', 0):.2%}\n"
                                f"- Hater flip rate: {flip_rates.get('hater', 0):.2%}\n\n"
                                f"**Probability Scores:**\n"
                                f"- p0 (original consistency): {confidence_result.get('p0', 0):.4f}\n"
                                f"- p1 (contrarian robustness): {confidence_result.get('p1', 0):.4f}\n"
                                f"- p2 (deceiver robustness): {confidence_result.get('p2', 0):.4f}\n"
                                f"- p3 (hater robustness): {confidence_result.get('p3', 0):.4f}\n\n"
                                f"---\n\n"
                                f"{original_summary}"
                                f"{adversarial_summary}"
                                f"{dist_summary}\n"
                                f"See detailed LLM interactions above for all prompts and responses."
                            )
                except Exception as e:
                    logger.warning(f"  -> Confidence estimation failed: {e}, using default 0.5")
                    confidence_score = 0.5
                    confidence_result = None
                
                expert_results_with_confidence.append({
                    "expert": expert_type,
                    "label": initial_label,
                    "confidence": confidence_score,
                    "prompt": full_prompt,
                    "response": result.get("response", ""),
                    "confidence_details": confidence_result,
                    "question_type": question_type
                })
                
            except Exception as e:
                logger.warning(f"Expert {expert_type} failed: {e}")
                continue
        
        # Check if any experts succeeded
        if not expert_results_with_confidence:
            logger.warning("All experts failed, returning default result")
            return {
                "label": 0,
                "confidence": 0.0,
                "expert_results": [],
                "aggregation_method": "none"
            }
        
        # Final aggregation considering all expert results with their confidence
        final_result = self._aggregate_experts_with_confidence(expert_results_with_confidence)
        
        logger.info(f"Final aggregated result: label={final_result['label']}, confidence={final_result['confidence']:.4f}")
        
        # Add final aggregation section to report
        if get_report_logger is not None:
            report_logger = get_report_logger()
            if report_logger.enabled and report_logger.report_file is not None:
                final_label_text = "Yes (1)" if final_result['label'] == 1 else "No (0)"
                details = final_result.get('details', {})
                
                expert_summary = "\n".join([
                    f"- **{r['expert']}**: label={r['label']}, confidence={r['confidence']:.4f}"
                    for r in expert_results_with_confidence
                ])
                
                report_logger.add_section(
                    f"Final Aggregated Result",
                    f"**Aggregation Method:** {final_result['method']}\n\n"
                    f"**Final Label:** {final_label_text}\n\n"
                    f"**Final Confidence:** {final_result['confidence']:.4f}\n\n"
                    f"**Expert Results:**\n{expert_summary}\n\n"
                    f"**Voting Details:**\n"
                    f"- Label vote counts: {details.get('label_vote_counts', {})}\n"
                    f"- Winning experts: {', '.join(details.get('winning_experts', []))}\n"
                    f"- Winning expert count: {details.get('winning_expert_count', 0)} / {details.get('total_expert_count', 0)}\n"
                )
        
        return {
            "label": final_result["label"],
            "confidence": final_result["confidence"],
            "expert_results": expert_results_with_confidence,
            "aggregation_method": final_result["method"],
            "aggregation_details": final_result.get("details", {})
        }
    
    def check_backdoor_path(self, x1: str, x2: str) -> Dict[str, Any]:
        """Check if backdoor path exists between x1 and x2."""
        logger.info(f"Checking backdoor path: {x1} ↔ {x2}")
        return self._run_expert_with_confidence(
            BackdoorPathExpert,
            "backdoor_path",
            x1,
            x2
        )
    
    def check_independence(self, x1: str, x2: str, after_blocking: bool = False) -> Dict[str, Any]:
        """Check if x1 and x2 are independent (optionally after blocking backdoor paths)."""
        logger.info(f"Checking independence: {x1} ⊥ {x2} (after_blocking={after_blocking})")
        return self._run_expert_with_confidence(
            IndependenceExpert,
            "independence",
            x1,
            x2
        )
    
    def check_latent_confounder(self, x1: str, x2: str, after_blocking: bool = False) -> Dict[str, Any]:
        """Check if latent confounder exists between x1 and x2."""
        logger.info(f"Checking latent confounder: {x1} ← ? → {x2} (after_blocking={after_blocking})")
        return self._run_expert_with_confidence(
            LatentConfounderExpert,
            "latent_confounder",
            x1,
            x2
        )
    
    def check_causal_direction(self, x1: str, x2: str, after_blocking: bool = False) -> Dict[str, Any]:
        """Check if x1 causes x2."""
        logger.info(f"Checking causal direction: {x1} → {x2} (after_blocking={after_blocking})")
        return self._run_expert_with_confidence(
            CausalDirectionExpert,
            "causal_direction",
            x1,
            x2
        )
    
    def tree_query(self, x1: str, x2: str) -> Dict[str, Any]:
        """
        Perform tree-based causal query with confidence-based branching.
        
        The algorithm explores branches based on confidence thresholds:
        - If confidence >= trust_confidence: stop and record result
        - If confidence < trust_confidence: continue exploring other branches
        
        Decision tree:
        1. Check backdoor path
           - If confidence is high enough, explore only the matching branch
           - Otherwise, explore both branches (with and without backdoor assumption)
        2. For each branch:
           a. Check independence
           b. Check latent confounder  
           c. Check causal direction
        
        Returns:
            Dict with:
                - "have_backdoor": bool indicating backdoor path existence
                - "backdoor_confidence": confidence score for backdoor detection
                - "backdoor_log": execution log for backdoor check
                - "results": list of dicts, each containing "relation", "confidence", "log"
        """
        logger.info(f"=== Tree Query: {x1} vs {x2} ===")
        
        # Result structure
        query_results = {
            "have_backdoor": None,
            "backdoor_confidence": None,
            "backdoor_log": [],
            "results": []
        }
        
        # Step 1: Check backdoor path
        res_backdoor = self.check_backdoor_path(x1, x2)
        query_results["have_backdoor"] = res_backdoor["label"] == 1
        query_results["backdoor_confidence"] = res_backdoor["confidence"]
        query_results["backdoor_log"] = [("backdoor_path", res_backdoor)]
        
        backdoor_confident = res_backdoor["confidence"] >= self.trust_confidence
        
        # Determine which branches to explore
        explore_with_backdoor = res_backdoor["label"] == 1 or not backdoor_confident
        explore_without_backdoor = res_backdoor["label"] == 0 or not backdoor_confident
        
        if backdoor_confident:
            if res_backdoor["label"] == 1:
                logger.info(f"Backdoor path detected with high confidence ({res_backdoor['confidence']:.4f}) → analyzing after blocking only")
            else:
                logger.info(f"No backdoor path with high confidence ({res_backdoor['confidence']:.4f}) → direct analysis only")
        else:
            logger.info(f"Backdoor confidence ({res_backdoor['confidence']:.4f}) below threshold ({self.trust_confidence}) → exploring both branches")
        
        # Branch 1: With backdoor (after blocking)
        if explore_with_backdoor:
            self._explore_branch(
                x1, x2,
                after_blocking=True,
                branch_name="after_block",
                query_results=query_results
            )
        
        # Branch 2: Without backdoor (direct analysis)
        if explore_without_backdoor:
            self._explore_branch(
                x1, x2,
                after_blocking=False,
                branch_name="no_backdoor",
                query_results=query_results
            )
        
        return query_results
    
    def tree_query_all(self, x1: str, x2: str) -> Dict[str, Any]:
        """
        Perform tree-based causal query exploring ALL branches unconditionally.
        
        Unlike tree_query(), this method:
        - Always explores BOTH branches (with and without backdoor assumption)
        - Never stops early based on confidence thresholds
        - Executes all checks (independence, latent confounder, causal direction)
        
        This is useful for comprehensive analysis where you want to see all
        possible results regardless of confidence levels.
        
        Returns:
            Dict with same format as tree_query():
                - "have_backdoor": bool indicating backdoor path existence
                - "backdoor_confidence": confidence score for backdoor detection
                - "backdoor_log": execution log for backdoor check
                - "results": list of dicts, each containing "relation", "confidence", "log"
        """
        logger.info(f"=== Tree Query ALL (exhaustive): {x1} vs {x2} ===")
        
        # Result structure
        query_results = {
            "have_backdoor": None,
            "backdoor_confidence": None,
            "backdoor_log": [],
            "results": []
        }
        
        # Step 1: Check backdoor path (still needed for recording, but doesn't affect branching)
        res_backdoor = self.check_backdoor_path(x1, x2)
        query_results["have_backdoor"] = res_backdoor["label"] == 1
        query_results["backdoor_confidence"] = res_backdoor["confidence"]
        query_results["backdoor_log"] = [("backdoor_path", res_backdoor)]
        
        logger.info(f"Backdoor result: label={res_backdoor['label']}, confidence={res_backdoor['confidence']:.4f}")
        logger.info("Exploring ALL branches unconditionally...")
        
        # Branch 1: With backdoor assumption (after blocking)
        logger.info("--- Branch 1: After blocking backdoor paths ---")
        self._explore_branch_all(
            x1, x2,
            after_blocking=True,
            branch_name="after_block",
            query_results=query_results
        )
        
        # Branch 2: Without backdoor assumption (direct analysis)
        logger.info("--- Branch 2: Direct analysis (no backdoor) ---")
        self._explore_branch_all(
            x1, x2,
            after_blocking=False,
            branch_name="no_backdoor",
            query_results=query_results
        )
        
        logger.info(f"Tree query ALL complete. Total results: {len(query_results['results'])}")
        return query_results
    
    def _explore_branch_all(
        self,
        x1: str,
        x2: str,
        after_blocking: bool,
        branch_name: str,
        query_results: Dict[str, Any]
    ) -> None:
        """
        Explore a single branch exhaustively without early stopping.
        
        Unlike _explore_branch(), this method:
        - Never stops early based on confidence thresholds
        - Always executes ALL checks (independence, latent confounder, causal direction)
        - Records all results regardless of label values
        
        Args:
            x1: First variable
            x2: Second variable
            after_blocking: Whether to analyze after blocking backdoor paths
            branch_name: Name suffix for logging ("after_block" or "no_backdoor")
            query_results: Dictionary to append results to
        """
        execution_log = []
        
        # Step 2a: Check independence (always execute, never stop early)
        res_ind = self.check_independence(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"independence_{branch_name}", res_ind))
        
        if res_ind["label"] == 1:  # Independent
            query_results["results"].append({
                "relation": "independent",
                "confidence": res_ind["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_ind.get("expert_results", []),
                "aggregation_method": res_ind.get("aggregation_method", "unknown")
            })
            logger.info(f"Independent (confidence: {res_ind['confidence']:.4f})")
        else:  # Not independent
            query_results["results"].append({
                "relation": "not independent",
                "confidence": res_ind["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_ind.get("expert_results", []),
                "aggregation_method": res_ind.get("aggregation_method", "unknown")
            })
            logger.info(f"Not independent (confidence: {res_ind['confidence']:.4f})")
        
        # Step 2b: Check latent confounder (always execute, never stop early)
        res_latent = self.check_latent_confounder(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"latent_confounder_{branch_name}", res_latent))
        
        if res_latent["label"] == 1:  # Latent confounder exists
            query_results["results"].append({
                "relation": "<->",
                "confidence": res_latent["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_latent.get("expert_results", []),
                "aggregation_method": res_latent.get("aggregation_method", "unknown")
            })
            logger.info(f"Latent confounder exists (confidence: {res_latent['confidence']:.4f})")
        else:  # No latent confounder
            query_results["results"].append({
                "relation": "not <->",
                "confidence": res_latent["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_latent.get("expert_results", []),
                "aggregation_method": res_latent.get("aggregation_method", "unknown")
            })
            logger.info(f"No latent confounder (confidence: {res_latent['confidence']:.4f})")
        
        # Step 2c: Check causal direction (both directions, always execute)
        res_dir_xy = self.check_causal_direction(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"causal_direction_{branch_name}_x_to_y", res_dir_xy))
        
        res_dir_yx = self.check_causal_direction(x2, x1, after_blocking=after_blocking)
        execution_log.append((f"causal_direction_{branch_name}_y_to_x", res_dir_yx))
        
        # Record -> direction result
        if res_dir_xy["label"] == 1:
            query_results["results"].append({
                "relation": "->",
                "confidence": res_dir_xy["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_dir_xy.get("expert_results", []),
                "aggregation_method": res_dir_xy.get("aggregation_method", "unknown")
            })
            logger.info(f"-> supported (confidence: {res_dir_xy['confidence']:.4f})")
        else:
            query_results["results"].append({
                "relation": "not ->",
                "confidence": res_dir_xy["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_dir_xy.get("expert_results", []),
                "aggregation_method": res_dir_xy.get("aggregation_method", "unknown")
            })
            logger.info(f"-> not supported (confidence: {res_dir_xy['confidence']:.4f})")
        
        # Record <- direction result
        if res_dir_yx["label"] == 1:
            query_results["results"].append({
                "relation": "<-",
                "confidence": res_dir_yx["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_dir_yx.get("expert_results", []),
                "aggregation_method": res_dir_yx.get("aggregation_method", "unknown")
            })
            logger.info(f"<- supported (confidence: {res_dir_yx['confidence']:.4f})")
        else:
            query_results["results"].append({
                "relation": "not <-",
                "confidence": res_dir_yx["confidence"],
                "branch_name": branch_name,
                "log": execution_log.copy(),
                "expert_results": res_dir_yx.get("expert_results", []),
                "aggregation_method": res_dir_yx.get("aggregation_method", "unknown")
            })
            logger.info(f"<- not supported (confidence: {res_dir_yx['confidence']:.4f})")
    
    def filter_tree_query_all_results(
        self,
        query_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter tree_query_all() results to simulate tree_query() behavior.
        
        This method applies trust_confidence filtering to the comprehensive results
        from tree_query_all(), keeping only the results that would have been explored
        in tree_query() based on confidence thresholds.
        
        The filtering logic mirrors tree_query():
        1. If backdoor confidence >= trust_confidence, keep only the matching branch
        2. If backdoor confidence < trust_confidence, keep both branches
        3. Within each branch, stop at the first high-confidence result
        
        Args:
            query_results: Output from tree_query_all()
            
        Returns:
            Filtered results in the same format, simulating tree_query() behavior
        """
        logger.info("Filtering tree_query_all results with trust_confidence threshold")
        
        # Copy basic backdoor information
        filtered_results = {
            "have_backdoor": query_results.get("have_backdoor"),
            "backdoor_confidence": query_results.get("backdoor_confidence"),
            "backdoor_log": query_results.get("backdoor_log"),
            "results": []
        }
        
        backdoor_confidence = query_results.get("backdoor_confidence", 0.0)
        have_backdoor = query_results.get("have_backdoor", False)
        backdoor_confident = backdoor_confidence >= self.trust_confidence
        
        # Determine which branches should be kept
        keep_after_block = have_backdoor or not backdoor_confident
        keep_no_backdoor = not have_backdoor or not backdoor_confident
        
        if backdoor_confident:
            if have_backdoor:
                logger.info(f"Backdoor confident ({backdoor_confidence:.4f}) → keeping only 'after_block' branch")
            else:
                logger.info(f"No backdoor confident ({backdoor_confidence:.4f}) → keeping only 'no_backdoor' branch")
        else:
            logger.info(f"Backdoor not confident ({backdoor_confidence:.4f}) → keeping both branches")
        
        # Filter results by branch
        all_results = query_results.get("results", [])
        
        # Group results by branch
        after_block_results = []
        no_backdoor_results = []
        
        for result in all_results:
            log = result.get("log", [])
            if not log:
                continue
            
            # Determine branch by checking the log entries
            branch_name = None
            for step_name, _ in log:
                if "after_block" in step_name:
                    branch_name = "after_block"
                    break
                elif "no_backdoor" in step_name:
                    branch_name = "no_backdoor"
                    break
            
            if branch_name == "after_block":
                after_block_results.append(result)
            elif branch_name == "no_backdoor":
                no_backdoor_results.append(result)
        
        # Apply early stopping logic to each branch
        if keep_after_block:
            filtered_after_block = self._apply_early_stopping(after_block_results, "after_block")
            filtered_results["results"].extend(filtered_after_block)
        
        if keep_no_backdoor:
            filtered_no_backdoor = self._apply_early_stopping(no_backdoor_results, "no_backdoor")
            filtered_results["results"].extend(filtered_no_backdoor)
        
        logger.info(f"Filtered results: {len(filtered_results['results'])}/{len(all_results)} kept")
        return filtered_results
    
    def _apply_early_stopping(
        self,
        branch_results: List[Dict[str, Any]],
        branch_name: str
    ) -> List[Dict[str, Any]]:
        """
        Apply early stopping logic to a branch's results.
        
        Simulates the behavior of _explore_branch():
        - Keep results in order until a high-confidence valid relation is found
        - Stop after independence or latent confounder with high confidence
        - Always include causal direction checks
        
        Args:
            branch_results: List of results from one branch
            branch_name: "after_block" or "no_backdoor"
            
        Returns:
            Filtered list of results with early stopping applied
        """
        if not branch_results:
            return []
        
        filtered = []
        
        # Order: independence → latent confounder → causal directions
        # Find each check's results
        independence_results = [r for r in branch_results if r["relation"] in ["independent", "not independent"]]
        latent_results = [r for r in branch_results if r["relation"] in ["<->", "not <->"]]
        direction_results = [r for r in branch_results if r["relation"] in ["->", "<-", "not ->", "not <-"]]
        
        # Step 1: Independence check
        if independence_results:
            ind_result = independence_results[0]
            filtered.append(ind_result)
            
            # If independent with high confidence, stop here
            if ind_result["relation"] == "independent" and ind_result["confidence"] >= self.trust_confidence:
                logger.debug(f"Branch {branch_name}: stopping after high-confidence independence")
                return filtered
        
        # Step 2: Latent confounder check
        if latent_results:
            lat_result = latent_results[0]
            filtered.append(lat_result)
            
            # If latent confounder exists with high confidence, stop here
            if lat_result["relation"] == "<->" and lat_result["confidence"] >= self.trust_confidence:
                logger.debug(f"Branch {branch_name}: stopping after high-confidence latent confounder")
                return filtered
        
        # Step 3: Causal direction checks (always include)
        # In _explore_branch, directions are always checked if we get here
        # Add the results that would actually be recorded
        for dir_result in direction_results:
            relation = dir_result["relation"]
            
            # Only include positive relations or the selected negative one
            if relation in ["->", "<-"]:
                filtered.append(dir_result)
        
        # If no positive direction, add the one with lower rejection confidence
        # (matching the logic in _explore_branch)
        positive_directions = [r for r in filtered if r["relation"] in ["->", "<-"]]
        if not positive_directions and direction_results:
            not_xy = next((r for r in direction_results if r["relation"] == "not ->"), None)
            not_yx = next((r for r in direction_results if r["relation"] == "not <-"), None)
            
            if not_xy and not_yx:
                # Choose the one with lower confidence (less confident rejection)
                if not_xy["confidence"] <= not_yx["confidence"]:
                    # Invert confidence like in _explore_branch
                    filtered.append({
                        "relation": "->",
                        "confidence": 1.0 - not_xy["confidence"],
                        "log": not_xy["log"]
                    })
                else:
                    filtered.append({
                        "relation": "<-",
                        "confidence": 1.0 - not_yx["confidence"],
                        "log": not_yx["log"]
                    })
        
        return filtered

    def _explore_branch(
        self,
        x1: str,
        x2: str,
        after_blocking: bool,
        branch_name: str,
        query_results: Dict[str, Any]
    ) -> None:
        """
        Explore a single branch of the causal discovery tree.
        
        This method checks independence, latent confounder, and causal direction
        based on confidence thresholds. Results are appended to query_results.
        
        Args:
            x1: First variable
            x2: Second variable
            after_blocking: Whether to analyze after blocking backdoor paths
            branch_name: Name suffix for logging ("after_block" or "no_backdoor")
            query_results: Dictionary to append results to
        """
        execution_log = []
        
        # Step 2a: Check independence
        res_ind = self.check_independence(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"independence_{branch_name}", res_ind))
        
        if res_ind["label"] == 1:  # Independent
            query_results["results"].append({
                "relation": "independent",
                "confidence": res_ind["confidence"],
                "log": execution_log.copy()
            })
            if res_ind["confidence"] >= self.trust_confidence:
                logger.info(f"Independence detected with high confidence ({res_ind['confidence']:.4f}) → stopping branch")
                return
        else:  # Not independent - also record this
            query_results["results"].append({
                "relation": "not independent",
                "confidence": res_ind["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"Not independent (confidence: {res_ind['confidence']:.4f}) → continuing to check latent confounder")
        
        # Step 2b: Check latent confounder
        res_latent = self.check_latent_confounder(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"latent_confounder_{branch_name}", res_latent))
        
        if res_latent["label"] == 1:  # Latent confounder exists
            query_results["results"].append({
                "relation": "<->",
                "confidence": res_latent["confidence"],
                "log": execution_log.copy()
            })
            if res_latent["confidence"] >= self.trust_confidence:
                logger.info(f"Latent confounder detected with high confidence ({res_latent['confidence']:.4f}) → stopping branch")
                return
        else:  # No latent confounder - record
            query_results["results"].append({
                "relation": "not <->",
                "confidence": res_latent["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"No latent confounder (confidence: {res_latent['confidence']:.4f}) → continuing to check causal direction")
        
        # Step 2c: Check causal direction (both directions)
        res_dir_xy = self.check_causal_direction(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"causal_direction_{branch_name}_x_to_y", res_dir_xy))
        
        res_dir_yx = self.check_causal_direction(x2, x1, after_blocking=after_blocking)
        execution_log.append((f"causal_direction_{branch_name}_y_to_x", res_dir_yx))
        
        # Record both direction results separately
        # For -> direction
        if res_dir_xy["label"] == 1:
            query_results["results"].append({
                "relation": "->",
                "confidence": res_dir_xy["confidence"],
                "log": execution_log.copy()
            })
        
        # For <- direction
        if res_dir_yx["label"] == 1:
            query_results["results"].append({
                "relation": "<-",
                "confidence": res_dir_yx["confidence"],
                "log": execution_log.copy()
            })
        
        # If neither direction is supported, record the one with lower rejection confidence
        if res_dir_xy["label"] == 0 and res_dir_yx["label"] == 0:
            # Both rejected - choose the one rejected with less confidence
            if res_dir_xy["confidence"] <= res_dir_yx["confidence"]:
                query_results["results"].append({
                    "relation": "->",
                    "confidence": 1.0 - res_dir_xy["confidence"],  # Invert confidence for rejected
                    "log": execution_log.copy()
                })
            else:
                query_results["results"].append({
                    "relation": "<-",
                    "confidence": 1.0 - res_dir_yx["confidence"],  # Invert confidence for rejected
                    "log": execution_log.copy()
                })
    
    def resolve_query_results(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve query results by selecting the relation with highest confidence.
        
        Args:
            query_results: Output from tree_query(), containing:
                - "have_backdoor": bool
                - "backdoor_confidence": float
                - "backdoor_log": list
                - "results": list of dicts with "relation", "confidence", "log"
        
        Returns:
            Dict with:
                - "relation": the final causal relationship (highest confidence)
                - "confidence": the confidence score of the selected relation
                - "have_backdoor": backdoor path existence
                - "backdoor_confidence": confidence for backdoor detection
                - "all_results": all explored results for reference
        """
        results = query_results.get("results", [])
        
        # Define valid causal relationships (exclude negations like "not independent", "not <->")
        valid_relations = {"independent", "<->", "->", "<-"}
        
        # Filter to only valid relations for resolution
        valid_results = [r for r in results if r.get("relation") in valid_relations]
        
        if not valid_results:
            logger.warning("No valid causal relations found, returning default")
            return {
                "relation": "unknown",
                "confidence": 0.0,
                "have_backdoor": query_results.get("have_backdoor"),
                "backdoor_confidence": query_results.get("backdoor_confidence"),
                "all_results": results  # Keep all results for reference
            }
        
        # Find result with highest confidence among valid relations only
        best_result = max(valid_results, key=lambda r: r["confidence"])
        
        logger.info(f"Resolved relation: {best_result['relation']} (confidence: {best_result['confidence']:.4f})")
        logger.info(f"Valid results: {len(valid_results)}/{len(results)} total explored")
        
        return {
            "relation": best_result["relation"],
            "confidence": best_result["confidence"],
            "have_backdoor": query_results.get("have_backdoor"),
            "backdoor_confidence": query_results.get("backdoor_confidence"),
            "all_results": results
        }
    
    def discover_all_relations(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Discover causal relations between all variable pairs.
        
        Returns:
            Dictionary mapping variable pairs to their causal relationships.
            Each entry contains the resolved result with "relation", "confidence",
            "have_backdoor", "backdoor_confidence", and "all_results".
        """
        logger.info(f"=== Discovering relations for all {len(self.all_variables)} variables ===")
        all_relations = {}
        
        for x1, x2 in combinations(self.all_variables, 2):
            logger.info(f"\nAnalyzing pair: ({x1}, {x2})")
            query_results = self.tree_query(x1, x2)
            resolved = self.resolve_query_results(query_results)
            all_relations[(x1, x2)] = resolved
            logger.info(f"Result: {resolved['relation']} (confidence: {resolved['confidence']:.4f})")
        
        return all_relations
    
    def create_prior_causal_graph(
        self,
        all_relations: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], str]:
        """
        Generate prior causal graph directly from discovered relations.
        
        Simply uses the relation with highest confidence from each pair's results.
        
        Args:
            all_relations: Output from discover_all_relations()
            
        Returns:
            Prior graph mapping pairs to relation types
        """
        logger.info("Creating prior causal graph")
        prior_graph = {}
        
        for pair, result in all_relations.items():
            relation = result.get("relation", "unknown")
            confidence = result.get("confidence", 0.0)
            
            # Directly use the resolved relation
            if relation == "independent":
                prior_graph[pair] = "independent"
            elif relation == "<->":
                prior_graph[pair] = "<->"
            elif relation == "->":
                prior_graph[pair] = "->"
            elif relation == "<-":
                prior_graph[pair] = "<-"
            else:
                prior_graph[pair] = "unknown"
            
            logger.debug(f"{pair}: {prior_graph[pair]} (confidence={confidence:.3f})")
        
        return prior_graph
