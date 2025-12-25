"""
Benchmark Runner for Causal Discovery

This module runs tree_query_all() on each variable pair in a SimpleCausalBenchmark
and stores all possible results with their confidence scores in a CSV file.

The CSV format:
var1, var2, relation, confidence, branch_name, [additional metadata columns]
"""

import csv
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from itertools import combinations

from causal_benchmark import SimpleCausalBenchmark

# Use TYPE_CHECKING to avoid circular imports and runtime issues
if TYPE_CHECKING:
    from tree_query.causal_discovery_framework import CausalDiscoveryFramework

# Try to import framework and logging, provide helpful error messages
try:
    from tree_query.causal_discovery_framework import CausalDiscoveryFramework
except ImportError as e:
    CausalDiscoveryFramework = None
    _FRAMEWORK_IMPORT_ERROR = str(e)

try:
    from llm_utils.logging_config import setup_logging
except ImportError:
    # Fallback: use basic logging setup
    def setup_logging(level="INFO", **kwargs):
        """Fallback logging setup when llm_utils is not available."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%m-%d %H:%M:%S'
        )
        return logging.getLogger()

# Configure module logger
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for executing causal discovery on benchmark datasets.
    
    This class orchestrates the execution of tree_query_all() on all variable
    pairs in a benchmark and saves the results to CSV for later analysis.
    """
    
    def __init__(
        self,
        framework: Any,
        output_dir: str = "benchmark_results",
        seed: int = 42
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            framework: CausalDiscoveryFramework instance to use for queries
            output_dir: Directory to store output CSV files
            seed: Random seed for reproducibility
        
        Raises:
            ImportError: If CausalDiscoveryFramework could not be imported
        """
        if CausalDiscoveryFramework is None:
            raise ImportError(
                f"Cannot import CausalDiscoveryFramework. "
                f"Original error: {_FRAMEWORK_IMPORT_ERROR}"
            )
        
        self.framework = framework
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Storage for detailed Yes/No results
        self.yes_no_data = []
        
        logger.info(f"Initialized BenchmarkRunner with output_dir: {self.output_dir}")
    
    def _parse_relation_from_result(self, result: Dict[str, Any]) -> str:
        """
        Parse the relation string from a result dictionary.
        
        Args:
            result: Result dict from tree_query_all containing 'relation' key
        
        Returns:
            Relation string: '->', '<-', '<->', 'independent', or negations
        """
        relation = result.get("relation", "unknown")
        
        # Return relation as-is since we now use the new format
        # All relations: '->', '<-', '<->', 'independent', 'not independent', 
        # 'not ->', 'not <-', 'not <->'
        return relation
    
    def _collect_yes_no_results(
        self,
        var1: str,
        var2: str,
        query_result: Dict[str, Any]
    ):
        """
        Collect detailed Yes/No results from all experts and samples.
        
        Args:
            var1: First variable
            var2: Second variable
            query_result: Complete result from tree_query_all
        """
        # Process each result branch
        for result in query_result.get("results", []):
            branch_name = result.get("branch_name", "unknown")
            
            # Extract expert results with confidence details
            expert_results = result.get("expert_results", [])
            
            for expert_result in expert_results:
                expert_type = expert_result.get("expert", "unknown")
                expert_label = expert_result.get("label", 0)
                expert_confidence = expert_result.get("confidence", 0.0)
                question_type = expert_result.get("question_type", "unknown")
                
                confidence_details = expert_result.get("confidence_details")
                if not confidence_details:
                    continue
                
                # Collect original samples
                original_answers = confidence_details.get("original_answers", [])
                for i, orig_answer in enumerate(original_answers):
                    self.yes_no_data.append({
                        "var1": var1,
                        "var2": var2,
                        "branch_name": branch_name,
                        "expert_type": expert_type,
                        "question_type": question_type,
                        "sample_type": "original",
                        "sample_index": i,
                        "adversarial_type": "none",
                        "label": orig_answer.label,
                        "result": "Yes" if orig_answer.label == 1 else "No",
                        "expert_initial_label": expert_label,
                        "expert_confidence": f"{expert_confidence:.6f}"
                    })
                
                # Collect adversarial samples
                adversarial_samples = confidence_details.get("adversarial_samples", [])
                for adv_sample in adversarial_samples:
                    self.yes_no_data.append({
                        "var1": var1,
                        "var2": var2,
                        "branch_name": branch_name,
                        "expert_type": expert_type,
                        "question_type": question_type,
                        "sample_type": "adversarial",
                        "sample_index": adv_sample.set_index,
                        "adversarial_type": adv_sample.adversarial_type,
                        "label": adv_sample.label,
                        "result": "Yes" if adv_sample.label == 1 else "No",
                        "original_label": adv_sample.original_label,
                        "flipped": "Yes" if adv_sample.label != adv_sample.original_label else "No",
                        "expert_initial_label": expert_label,
                        "expert_confidence": f"{expert_confidence:.6f}"
                    })
    
    def export_yes_no_results(self, output_filename: str) -> str:
        """
        Export all collected Yes/No results to a CSV file.
        
        Args:
            output_filename: Name for output CSV file
        
        Returns:
            Path to the generated CSV file
        """
        output_path = self.output_dir / output_filename
        
        if not self.yes_no_data:
            logger.warning("No Yes/No data to export")
            return str(output_path)
        
        logger.info(f"Exporting {len(self.yes_no_data)} Yes/No results to: {output_path}")
        
        # Determine headers based on data (original vs adversarial samples have different fields)
        headers = [
            "var1", "var2", "branch_name", "expert_type", "question_type",
            "sample_type", "sample_index", "adversarial_type",
            "label", "result"
        ]
        
        # Check if we have adversarial samples to add extra columns
        has_adversarial = any(row.get("sample_type") == "adversarial" for row in self.yes_no_data)
        if has_adversarial:
            headers.extend(["original_label", "flipped"])
        
        headers.extend(["expert_initial_label", "expert_confidence"])
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.yes_no_data)
        
        logger.info(f"Yes/No results exported successfully")
        return str(output_path)
    
    def run_on_benchmark(
        self,
        benchmark: SimpleCausalBenchmark,
        output_filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Run tree_query_all() on all variable pairs in the benchmark.
        
        Args:
            benchmark: SimpleCausalBenchmark to run queries on
            output_filename: Name for output CSV file. If None, auto-generated
                           with timestamp
            include_metadata: Whether to include detailed metadata columns
        
        Returns:
            Path to the generated CSV file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_results_{timestamp}.csv"
        
        output_path = self.output_dir / output_filename
        
        logger.info(f"Starting benchmark run on {len(benchmark.all_variables)} variables")
        logger.info(f"Output file: {output_path}")
        
        # Prepare CSV headers
        headers = [
            "var1", "var2", "relation", "confidence",
            "branch_name", "branch_type"
        ]
        if include_metadata:
            headers.extend([
                "backdoor_exists", "backdoor_confidence",
                "query_timestamp"
            ])
        
        # Open CSV file for writing
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            # Iterate through all variable pairs
            total_pairs = len(list(combinations(benchmark.all_variables, 2)))
            processed = 0
            
            for var1, var2 in combinations(benchmark.all_variables, 2):
                processed += 1
                logger.info(f"Processing pair {processed}/{total_pairs}: ({var1}, {var2})")
                
                try:
                    # Run tree_query_all to get all possible results
                    query_result = self.framework.tree_query_all(var1, var2)
                    
                    # Collect Yes/No results from expert details
                    self._collect_yes_no_results(var1, var2, query_result)
                    
                    # Extract backdoor information
                    backdoor_exists = query_result.get("have_backdoor", False)
                    backdoor_confidence = query_result.get("backdoor_confidence", 0.0)
                    
                    # Process each result in the query
                    for result in query_result.get("results", []):
                        relation = self._parse_relation_from_result(result)
                        confidence = result.get("confidence", 0.0)
                        branch_name = result.get("branch_name", "unknown")
                        
                        # Determine branch type
                        if branch_name == "after_block":
                            branch_type = "with_backdoor"
                        elif branch_name == "no_backdoor":
                            branch_type = "without_backdoor"
                        else:
                            branch_type = "unknown"
                        
                        # Prepare row
                        row = {
                            "var1": var1,
                            "var2": var2,
                            "relation": relation,
                            "confidence": f"{confidence:.6f}",
                            "branch_name": branch_name,
                            "branch_type": branch_type
                        }
                        
                        if include_metadata:
                            row.update({
                                "backdoor_exists": str(backdoor_exists),
                                "backdoor_confidence": f"{backdoor_confidence:.6f}",
                                "query_timestamp": datetime.now().isoformat()
                            })
                        
                        writer.writerow(row)
                    
                    logger.info(f"  -> Generated {len(query_result.get('results', []))} results")
                    
                except Exception as e:
                    logger.error(f"Error processing pair ({var1}, {var2}): {e}", exc_info=True)
                    # Write error entry
                    error_row = {
                        "var1": var1,
                        "var2": var2,
                        "relation": "error",
                        "confidence": "0.0",
                        "branch_name": "error",
                        "branch_type": "error"
                    }
                    if include_metadata:
                        error_row.update({
                            "backdoor_exists": "False",
                            "backdoor_confidence": "0.0",
                            "query_timestamp": datetime.now().isoformat()
                        })
                    writer.writerow(error_row)
        
        logger.info(f"Benchmark run complete. Results saved to: {output_path}")
        return str(output_path)
    
    def run_on_variable_pairs(
        self,
        variable_pairs: List[tuple],
        output_filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Run tree_query_all() on a specific list of variable pairs.
        
        Args:
            variable_pairs: List of (var1, var2) tuples to query
            output_filename: Name for output CSV file
            include_metadata: Whether to include detailed metadata columns
        
        Returns:
            Path to the generated CSV file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"pairs_results_{timestamp}.csv"
        
        output_path = self.output_dir / output_filename
        
        logger.info(f"Starting run on {len(variable_pairs)} variable pairs")
        logger.info(f"Output file: {output_path}")
        
        # Prepare CSV headers
        headers = [
            "var1", "var2", "relation", "confidence",
            "branch_name", "branch_type"
        ]
        if include_metadata:
            headers.extend([
                "backdoor_exists", "backdoor_confidence",
                "query_timestamp"
            ])
        
        # Open CSV file for writing
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            processed = 0
            for var1, var2 in variable_pairs:
                processed += 1
                logger.info(f"Processing pair {processed}/{len(variable_pairs)}: ({var1}, {var2})")
                
                try:
                    # Run tree_query_all
                    query_result = self.framework.tree_query_all(var1, var2)
                    
                    backdoor_exists = query_result.get("have_backdoor", False)
                    backdoor_confidence = query_result.get("backdoor_confidence", 0.0)
                    
                    for result in query_result.get("results", []):
                        relation = self._parse_relation_from_result(result)
                        confidence = result.get("confidence", 0.0)
                        branch_name = result.get("branch_name", "unknown")
                        
                        if branch_name == "after_block":
                            branch_type = "with_backdoor"
                        elif branch_name == "no_backdoor":
                            branch_type = "without_backdoor"
                        else:
                            branch_type = "unknown"
                        
                        row = {
                            "var1": var1,
                            "var2": var2,
                            "relation": relation,
                            "confidence": f"{confidence:.6f}",
                            "branch_name": branch_name,
                            "branch_type": branch_type
                        }
                        
                        if include_metadata:
                            row.update({
                                "backdoor_exists": str(backdoor_exists),
                                "backdoor_confidence": f"{backdoor_confidence:.6f}",
                                "query_timestamp": datetime.now().isoformat()
                            })
                        
                        writer.writerow(row)
                    
                except Exception as e:
                    logger.error(f"Error processing pair ({var1}, {var2}): {e}", exc_info=True)
        
        logger.info(f"Run complete. Results saved to: {output_path}")
        return str(output_path)
