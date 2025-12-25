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
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            framework: CausalDiscoveryFramework instance to use for queries
            output_dir: Directory to store output CSV files
        
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
        
        logger.info(f"Initialized BenchmarkRunner with output_dir: {self.output_dir}")
    
    def _parse_relation_from_result(self, result: Dict[str, Any]) -> str:
        """
        Parse the relation string from a result dictionary.
        
        Args:
            result: Result dict from tree_query_all containing 'relation' key
        
        Returns:
            Relation string: 'x->y', 'y->x', 'x<->y', or 'independent'
        """
        relation = result.get("relation", "unknown")
        
        # Normalize relation format
        if relation == "independent" or relation == "no_relation":
            return "independent"
        elif relation == "x->y" or relation == "x_causes_y":
            return "x->y"
        elif relation == "y->x" or relation == "y_causes_x":
            return "y->x"
        elif relation == "x<->y" or relation == "bidirectional" or relation == "confounded":
            return "x<->y"
        else:
            logger.warning(f"Unknown relation format: {relation}, defaulting to 'independent'")
            return "independent"
    
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


def main():
    """Example usage of BenchmarkRunner with simulated data for testing."""
    # Setup logging
    setup_logging(level="INFO")
    
    logger.info("=" * 60)
    logger.info("BenchmarkRunner Example")
    logger.info("=" * 60)
    
    # This is a placeholder - in real usage, you would:
    # 1. Create a real LLM client
    # 2. Initialize CausalDiscoveryFramework
    # 3. Run on actual benchmark
    
    logger.info("\nNOTE: This example requires actual LLM client and framework.")
    logger.info("Use this module by importing BenchmarkRunner and providing:")
    logger.info("  - A configured CausalDiscoveryFramework instance")
    logger.info("  - A SimpleCausalBenchmark with ground truth relationships")
    logger.info("\nExample code:")
    logger.info("  from llm_utils import create_llm_client")
    logger.info("  from tree_query.causal_discovery_framework import CausalDiscoveryFramework")
    logger.info("  from causal_benchmark import SimpleCausalBenchmark")
    logger.info("  from run_benchmark import BenchmarkRunner")
    logger.info("")
    logger.info("  # Setup")
    logger.info("  client = create_llm_client(...)")
    logger.info("  framework = CausalDiscoveryFramework(client, all_variables=[...])")
    logger.info("  benchmark = SimpleCausalBenchmark([...])")
    logger.info("")
    logger.info("  # Run")
    logger.info("  runner = BenchmarkRunner(framework)")
    logger.info("  output_file = runner.run_on_benchmark(benchmark)")
    logger.info("  print(f'Results saved to: {output_file}')")


if __name__ == "__main__":
    main()
