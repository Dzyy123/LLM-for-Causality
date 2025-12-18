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
from typing import List, Dict, Any, Tuple, Union
from itertools import combinations

from llm_utils import LocalLLMClient, OnlineLLMClient
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
        k1_samples: int = 5,
        k2_samples: int = 2,
        seed: int = 42,
        max_workers: int = 10,
        trust_confidence: float = 0.75
    ):
        """
        Initialize the causal discovery framework.
        
        Args:
            client: LLM client for experts and confidence estimation
            all_variables: List of all variables in the causal system
            k1_samples: Number of original answer samples for confidence estimation
            k2_samples: Number of adversarial sets per original answer
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
        
        # Initialize adversary-based confidence estimator
        self.confidence_estimator = create_adversarial_confidence_estimator(
            client=client,
            k1_samples=k1_samples,
            k2_samples=k2_samples,
            seed=seed,
            max_workers=max_workers
        )
        
        logger.info(f"Initialized CausalDiscoveryFramework with {len(all_variables)} variables")
        logger.info(f"Confidence estimator: k1={k1_samples}, k2={k2_samples}")
        logger.info(f"Trust confidence threshold: {trust_confidence}")
    
    def _run_expert_with_confidence(
        self,
        expert_class: type,
        question_type: str,
        x1: str,
        x2: str
    ) -> Dict[str, Any]:
        """
        Run expert judgment with adversary-based confidence estimation.
        
        Args:
            expert_class: The expert class to use
            question_type: Type of question for routing
            x1: First variable
            x2: Second variable
            
        Returns:
            Dict with label, confidence score, and expert details
        """
        # Select experts via router
        selected_expert_types = self.router.select_experts(question_type, x1, x2, top_k=3)
        logger.info(f"Selected experts for '{question_type}': {selected_expert_types}")
        
        # Execute expert judgments
        expert_results = []
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
                expert_results.append({
                    "expert": expert_type,
                    "label": result["label"],
                    "prompt": full_prompt,  # Store the full prompt sent to LLM
                    "response": result.get("response", "")
                })
                logger.info(f"Expert {expert_type}: label={result['label']}")
                
            except Exception as e:
                logger.warning(f"Expert {expert_type} failed: {e}")
                continue
        
        # Aggregate expert opinions
        if not expert_results:
            logger.warning("All experts failed, returning default result")
            return {"label": 0, "confidence": 0.0, "expert_results": []}
        
        aggregated = aggregate_expert_results(expert_results)
        final_label = aggregated["label"]
        
        # Estimate confidence using adversary-based approach
        # Use the first successful expert for confidence estimation
        representative_expert = expert_class(
            base_prompt="",
            x1=x1,
            x2=x2,
            client=self.client,
            all_variables=self.all_variables,
            expert_type=selected_expert_types[0],
            seed=self.seed
        )
        representative_expert.base_prompt = representative_expert.generate_question()
        
        try:
            confidence_result = self.confidence_estimator.estimate_confidence(
                representative_expert,
                final_label
            )
            confidence_score = confidence_result["confidence_score"]
            logger.info(f"Confidence score: {confidence_score:.4f}")
        except Exception as e:
            logger.warning(f"Confidence estimation failed: {e}, using default 0.5")
            confidence_score = 0.5
            confidence_result = None
        
        return {
            "label": final_label,
            "confidence": confidence_score,
            "expert_results": expert_results,
            "confidence_details": confidence_result  # Include full adversarial data
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
                "log": execution_log.copy()
            })
            logger.info(f"Independent (confidence: {res_ind['confidence']:.4f})")
        else:  # Not independent
            query_results["results"].append({
                "relation": "not independent",
                "confidence": res_ind["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"Not independent (confidence: {res_ind['confidence']:.4f})")
        
        # Step 2b: Check latent confounder (always execute, never stop early)
        res_latent = self.check_latent_confounder(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"latent_confounder_{branch_name}", res_latent))
        
        if res_latent["label"] == 1:  # Latent confounder exists
            query_results["results"].append({
                "relation": "x<->y",
                "confidence": res_latent["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"Latent confounder exists (confidence: {res_latent['confidence']:.4f})")
        else:  # No latent confounder
            query_results["results"].append({
                "relation": "not x<->y",
                "confidence": res_latent["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"No latent confounder (confidence: {res_latent['confidence']:.4f})")
        
        # Step 2c: Check causal direction (both directions, always execute)
        res_dir_xy = self.check_causal_direction(x1, x2, after_blocking=after_blocking)
        execution_log.append((f"causal_direction_{branch_name}_x_to_y", res_dir_xy))
        
        res_dir_yx = self.check_causal_direction(x2, x1, after_blocking=after_blocking)
        execution_log.append((f"causal_direction_{branch_name}_y_to_x", res_dir_yx))
        
        # Record x->y direction result
        if res_dir_xy["label"] == 1:
            query_results["results"].append({
                "relation": "x->y",
                "confidence": res_dir_xy["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"x->y supported (confidence: {res_dir_xy['confidence']:.4f})")
        else:
            query_results["results"].append({
                "relation": "not x->y",
                "confidence": res_dir_xy["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"x->y not supported (confidence: {res_dir_xy['confidence']:.4f})")
        
        # Record y->x direction result
        if res_dir_yx["label"] == 1:
            query_results["results"].append({
                "relation": "y->x",
                "confidence": res_dir_yx["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"y->x supported (confidence: {res_dir_yx['confidence']:.4f})")
        else:
            query_results["results"].append({
                "relation": "not y->x",
                "confidence": res_dir_yx["confidence"],
                "log": execution_log.copy()
            })
            logger.info(f"y->x not supported (confidence: {res_dir_yx['confidence']:.4f})")
    
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
        latent_results = [r for r in branch_results if r["relation"] in ["x<->y", "not x<->y"]]
        direction_results = [r for r in branch_results if r["relation"] in ["x->y", "y->x", "not x->y", "not y->x"]]
        
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
            if lat_result["relation"] == "x<->y" and lat_result["confidence"] >= self.trust_confidence:
                logger.debug(f"Branch {branch_name}: stopping after high-confidence latent confounder")
                return filtered
        
        # Step 3: Causal direction checks (always include)
        # In _explore_branch, directions are always checked if we get here
        # Add the results that would actually be recorded
        for dir_result in direction_results:
            relation = dir_result["relation"]
            
            # Only include positive relations or the selected negative one
            if relation in ["x->y", "y->x"]:
                filtered.append(dir_result)
        
        # If no positive direction, add the one with lower rejection confidence
        # (matching the logic in _explore_branch)
        positive_directions = [r for r in filtered if r["relation"] in ["x->y", "y->x"]]
        if not positive_directions and direction_results:
            not_xy = next((r for r in direction_results if r["relation"] == "not x->y"), None)
            not_yx = next((r for r in direction_results if r["relation"] == "not y->x"), None)
            
            if not_xy and not_yx:
                # Choose the one with lower confidence (less confident rejection)
                if not_xy["confidence"] <= not_yx["confidence"]:
                    # Invert confidence like in _explore_branch
                    filtered.append({
                        "relation": "x->y",
                        "confidence": 1.0 - not_xy["confidence"],
                        "log": not_xy["log"]
                    })
                else:
                    filtered.append({
                        "relation": "y->x",
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
                "relation": "x<->y",
                "confidence": res_latent["confidence"],
                "log": execution_log.copy()
            })
            if res_latent["confidence"] >= self.trust_confidence:
                logger.info(f"Latent confounder detected with high confidence ({res_latent['confidence']:.4f}) → stopping branch")
                return
        else:  # No latent confounder - record
            query_results["results"].append({
                "relation": "not x<->y",
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
        # For x->y direction
        if res_dir_xy["label"] == 1:
            query_results["results"].append({
                "relation": "x->y",
                "confidence": res_dir_xy["confidence"],
                "log": execution_log.copy()
            })
        
        # For y->x direction
        if res_dir_yx["label"] == 1:
            query_results["results"].append({
                "relation": "y->x",
                "confidence": res_dir_yx["confidence"],
                "log": execution_log.copy()
            })
        
        # If neither direction is supported, record the one with lower rejection confidence
        if res_dir_xy["label"] == 0 and res_dir_yx["label"] == 0:
            # Both rejected - choose the one rejected with less confidence
            if res_dir_xy["confidence"] <= res_dir_yx["confidence"]:
                query_results["results"].append({
                    "relation": "x->y",
                    "confidence": 1.0 - res_dir_xy["confidence"],  # Invert confidence for rejected
                    "log": execution_log.copy()
                })
            else:
                query_results["results"].append({
                    "relation": "y->x",
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
        
        # Define valid causal relationships (exclude negations like "not independent", "not x<->y")
        valid_relations = {"independent", "x<->y", "x->y", "y->x"}
        
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
            elif relation == "x<->y":
                prior_graph[pair] = "<->"
            elif relation == "x->y":
                prior_graph[pair] = "x->y"
            elif relation == "y->x":
                prior_graph[pair] = "y->x"
            else:
                prior_graph[pair] = "unknown"
            
            logger.debug(f"{pair}: {prior_graph[pair]} (confidence={confidence:.3f})")
        
        return prior_graph
