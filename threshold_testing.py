"""
Threshold Testing for Causal Discovery Confidence

This module loads CSV results and tests different
trust_confidence thresholds to find the optimal value that maximizes
agreement with the ground truth benchmark.

The algorithm:
1. Load all raw results (all branches/confidence scores) from CSV
2. For each threshold value:
   - Select the highest confidence result per variable pair that exceeds threshold
   - If no result exceeds threshold, use the highest confidence result anyway
   - Build a SimpleCausalBenchmark from selected results
   - Evaluate against ground truth benchmark
3. Report which threshold gives best performance
"""

import csv
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import numpy as np

from causal_benchmark import SimpleCausalBenchmark

# Try to import llm_utils logging, fallback to basic logging
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


class ThresholdTester:
    """
    Test different confidence thresholds to find optimal performance.
    
    This class loads results from a CSV file.
    and systematically tests different threshold values to determine which
    threshold produces results that best match the ground truth.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the threshold tester.
        
        Args:
            csv_path: Path to CSV file containing benchmark results
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.raw_results = self._load_csv()
        logger.info(f"Loaded {len(self.raw_results)} results from {csv_path}")
    
    def _load_csv(self) -> List[Dict[str, str]]:
        """
        Load results from CSV file.
        
        Returns:
            List of dictionaries, each representing a row in the CSV
        """
        results = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Skip error rows
                if row.get("relation") == "error":
                    continue
                
                # Convert confidence to float
                try:
                    row["confidence"] = float(row["confidence"])
                except (ValueError, KeyError):
                    logger.warning(f"Invalid confidence value in row: {row}")
                    continue
                
                results.append(row)
        
        return results
    
    def _group_results_by_pair(self) -> Dict[Tuple[str, str], List[Dict]]:
        """
        Group results by variable pair.
        
        Returns:
            Dict mapping (var1, var2) tuples to lists of results for that pair
        """
        grouped = defaultdict(list)
        
        for result in self.raw_results:
            var1 = result["var1"]
            var2 = result["var2"]
            grouped[(var1, var2)].append(result)
        
        logger.info(f"Grouped results into {len(grouped)} variable pairs")
        return grouped
    
    def _select_results_with_threshold(
        self,
        threshold: float,
        fallback_to_max: bool = True
    ) -> List[Tuple[str, str, str]]:
        """
        Select best result for each variable pair using given threshold.
        
        Strategy:
        1. For each variable pair, find all results with confidence >= threshold
        2. If any exist, choose the one with highest confidence
        3. If none exist and fallback_to_max=True, choose highest confidence result
        4. If none exist and fallback_to_max=False, skip this pair
        
        Args:
            threshold: Minimum confidence threshold (0.0 - 1.0)
            fallback_to_max: If True, use highest confidence when none meet threshold
        
        Returns:
            List of (var1, var2, relation) tuples
        """
        grouped = self._group_results_by_pair()
        selected = []
        
        for (var1, var2), results in grouped.items():
            # Filter results meeting threshold
            above_threshold = [r for r in results if r["confidence"] >= threshold]
            
            if above_threshold:
                # Select highest confidence among those meeting threshold
                best = max(above_threshold, key=lambda r: r["confidence"])
                selected.append((var1, var2, best["relation"]))
                logger.debug(
                    f"Pair ({var1}, {var2}): Selected {best['relation']} "
                    f"with confidence {best['confidence']:.4f} (>= {threshold})"
                )
            elif fallback_to_max:
                # Fallback: select highest confidence regardless of threshold
                best = max(results, key=lambda r: r["confidence"])
                selected.append((var1, var2, best["relation"]))
                logger.debug(
                    f"Pair ({var1}, {var2}): Fallback to {best['relation']} "
                    f"with confidence {best['confidence']:.4f} (< {threshold})"
                )
            else:
                logger.debug(f"Pair ({var1}, {var2}): No result meets threshold, skipped")
        
        return selected
    
    def test_threshold(
        self,
        ground_truth: SimpleCausalBenchmark,
        threshold: float,
        fallback_to_max: bool = True,
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        Test a specific threshold value against ground truth.
        
        Args:
            ground_truth: SimpleCausalBenchmark with correct relationships
            threshold: Confidence threshold to test
            fallback_to_max: Whether to use max confidence when none meet threshold
            detailed: If True, return detailed metrics; otherwise just accuracy
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Select results using threshold
        selected_relations = self._select_results_with_threshold(
            threshold,
            fallback_to_max
        )
        
        # Build benchmark from selected results
        predicted_benchmark = SimpleCausalBenchmark(
            selected_relations,
            all_variables=ground_truth.all_variables
        )
        
        # Evaluate against ground truth
        if detailed:
            metrics = ground_truth.evaluate_detailed(predicted_benchmark)
        else:
            accuracy = ground_truth.evaluate(predicted_benchmark)
            metrics = {"recall": accuracy}
        
        logger.info(
            f"Threshold {threshold:.2f}: "
            f"Selected {len(selected_relations)} pairs, "
            f"Recall: {metrics.get('recall', 0):.2%}"
        )
        
        return metrics
    
    def test_threshold_range(
        self,
        ground_truth: SimpleCausalBenchmark,
        threshold_min: float = 0.0,
        threshold_max: float = 1.0,
        num_steps: int = 21,
        fallback_to_max: bool = True,
        detailed: bool = False
    ) -> List[Dict[str, any]]:
        """
        Test a range of threshold values.
        
        Args:
            ground_truth: SimpleCausalBenchmark with correct relationships
            threshold_min: Minimum threshold to test
            threshold_max: Maximum threshold to test
            num_steps: Number of threshold values to test
            fallback_to_max: Whether to use max confidence fallback
            detailed: Whether to compute detailed metrics
        
        Returns:
            List of dicts, each containing threshold and corresponding metrics
        """
        thresholds = np.linspace(threshold_min, threshold_max, num_steps)
        results = []
        
        logger.info(f"Testing {num_steps} threshold values from {threshold_min} to {threshold_max}")
        
        for threshold in thresholds:
            metrics = self.test_threshold(
                ground_truth,
                threshold,
                fallback_to_max,
                detailed
            )
            
            result_entry = {"threshold": threshold}
            result_entry.update(metrics)
            results.append(result_entry)
        
        return results
    
    def find_best_threshold(
        self,
        ground_truth: SimpleCausalBenchmark,
        threshold_min: float = 0.0,
        threshold_max: float = 1.0,
        num_steps: int = 21,
        metric: str = "recall",
        fallback_to_max: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find the threshold that maximizes a given metric.
        
        Args:
            ground_truth: SimpleCausalBenchmark with correct relationships
            threshold_min: Minimum threshold to test
            threshold_max: Maximum threshold to test
            num_steps: Number of threshold values to test
            metric: Metric to optimize ('recall', 'precision', 'f1', 'accuracy')
            fallback_to_max: Whether to use max confidence fallback
        
        Returns:
            Tuple of (best_threshold, metrics_dict)
        """
        results = self.test_threshold_range(
            ground_truth,
            threshold_min,
            threshold_max,
            num_steps,
            fallback_to_max,
            detailed=True
        )
        
        # Find best threshold
        best_result = max(results, key=lambda r: r.get(metric, 0.0))
        best_threshold = best_result["threshold"]
        
        logger.info(f"Best threshold: {best_threshold:.4f} with {metric}={best_result[metric]:.4f}")
        
        return best_threshold, best_result
    
    def generate_report(
        self,
        ground_truth: SimpleCausalBenchmark,
        output_path: Optional[str] = None,
        threshold_min: float = 0.0,
        threshold_max: float = 1.0,
        num_steps: int = 21
    ) -> str:
        """
        Generate a comprehensive threshold testing report.
        
        Args:
            ground_truth: SimpleCausalBenchmark with correct relationships
            output_path: Path to save report CSV. If None, auto-generated
            threshold_min: Minimum threshold to test
            threshold_max: Maximum threshold to test
            num_steps: Number of threshold values to test
        
        Returns:
            Path to the generated report file
        """
        if output_path is None:
            csv_stem = self.csv_path.stem
            output_path = self.csv_path.parent / f"{csv_stem}_threshold_report.csv"
        
        output_path = Path(output_path)
        
        logger.info(f"Generating threshold report: {output_path}")
        
        # Run tests
        results = self.test_threshold_range(
            ground_truth,
            threshold_min,
            threshold_max,
            num_steps,
            fallback_to_max=True,
            detailed=True
        )
        
        # Write to CSV
        headers = [
            "threshold", "accuracy", "precision", "recall", "f1",
            "total_baseline", "total_predicted", "matching"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for result in results:
                row = {
                    "threshold": f"{result['threshold']:.4f}",
                    "accuracy": f"{result['accuracy']:.6f}",
                    "precision": f"{result['precision']:.6f}",
                    "recall": f"{result['recall']:.6f}",
                    "f1": f"{result['f1']:.6f}",
                    "total_baseline": result["total_baseline"],
                    "total_predicted": result["total_other"],
                    "matching": result["matching"]
                }
                writer.writerow(row)
        
        logger.info(f"Report saved to: {output_path}")
        
        # Find and log best thresholds for each metric
        for metric in ["accuracy", "precision", "recall", "f1"]:
            best = max(results, key=lambda r: r[metric])
            logger.info(
                f"Best {metric}: {best[metric]:.4f} at threshold {best['threshold']:.4f}"
            )
        
        return str(output_path)


def main():
    """Example usage with simulated test data."""
    # Setup logging
    setup_logging(level="INFO")
    
    logger.info("=" * 60)
    logger.info("ThresholdTester Example")
    logger.info("=" * 60)
    
    # Create test CSV file with simulated data
    test_csv = Path("test_threshold_data.csv")
    logger.info(f"\nCreating test data: {test_csv}")
    
    with open(test_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "var1", "var2", "relation", "confidence",
            "branch_name", "branch_type"
        ])
        writer.writeheader()
        
        # Simulate some results
        test_data = [
            ("A", "B", "->", 0.95, "no_backdoor", "without_backdoor"),
            ("A", "B", "independent", 0.30, "after_block", "with_backdoor"),
            ("B", "C", "->", 0.85, "no_backdoor", "without_backdoor"),
            ("B", "C", "<-", 0.40, "after_block", "with_backdoor"),
            ("A", "C", "independent", 0.90, "no_backdoor", "without_backdoor"),
            ("A", "C", "<->", 0.25, "after_block", "with_backdoor"),
        ]
        
        for var1, var2, rel, conf, branch, btype in test_data:
            writer.writerow({
                "var1": var1, "var2": var2, "relation": rel,
                "confidence": conf, "branch_name": branch, "branch_type": btype
            })
    
    # Create ground truth benchmark
    ground_truth = SimpleCausalBenchmark([
        ("A", "B", "->"),
        ("B", "C", "->"),
        ("A", "C", "independent"),
    ])
    
    logger.info(f"\nGround truth benchmark:")
    logger.info(f"  {len(ground_truth.relationships)} relationships")
    
    # Test thresholds
    logger.info("\n" + "=" * 60)
    logger.info("Testing Thresholds")
    logger.info("=" * 60)
    
    tester = ThresholdTester(str(test_csv))
    
    # Find best threshold
    best_threshold, best_metrics = tester.find_best_threshold(
        ground_truth,
        threshold_min=0.0,
        threshold_max=1.0,
        num_steps=11,
        metric="f1"
    )
    
    logger.info(f"\nBest threshold: {best_threshold:.2f}")
    logger.info(f"Metrics at best threshold:")
    for key, value in best_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Generate report
    report_path = tester.generate_report(ground_truth)
    logger.info(f"\nFull report saved to: {report_path}")
    
    # Clean up test files
    logger.info("\n" + "=" * 60)
    logger.info("Cleaning up test files")
    logger.info("=" * 60)
    
    test_csv.unlink()
    logger.info(f"Deleted: {test_csv}")
    
    Path(report_path).unlink()
    logger.info(f"Deleted: {report_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
