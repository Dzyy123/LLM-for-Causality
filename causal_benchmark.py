"""
Causal Benchmark Module

Provides SimpleCausalBenchmark class for managing and evaluating causal relationships.
"""

import csv
from typing import Iterable, Tuple, Set, Optional, Dict
from collections import defaultdict
from pathlib import Path


class SimpleCausalBenchmark:
    """
    A benchmark class for storing and evaluating causal relationships.
    
    Attributes:
        relationships: Set of (var1, var2, relationship) tuples
        all_variables: Set of all variables in the benchmark
    """
    
    VALID_RELATIONSHIPS = {'->', '<-', '<->', 'independent'}
    
    def __init__(
        self, 
        relationships: Iterable[Tuple[str, str, str]], 
        all_variables: Optional[Set[str]] = None
    ):
        """
        Initialize a SimpleCausalBenchmark.
        
        Args:
            relationships: Iterable of (var1, var2, relationship) tuples.
                          Relationship must be one of: '->', '<-', '<->', 'independent'
            all_variables: Optional set of all variables. If None, automatically extracted
                          from relationships.
        
        Raises:
            ValueError: If relationship format is invalid or relationship type is not recognized.
            TypeError: If input format is incorrect.
        """
        self.relationships: Set[Tuple[str, str, str]] = set()
        self._relationship_dict: Dict[Tuple[str, str], str] = {}
        
        # Validate and store relationships
        for item in relationships:
            self._validate_and_add_relationship(item)
        
        # Set or auto-generate all_variables
        if all_variables is not None:
            self.all_variables = set(all_variables)
            # Verify all variables in relationships are in all_variables
            extracted_vars = self._extract_variables_from_relationships()
            if not extracted_vars.issubset(self.all_variables):
                missing = extracted_vars - self.all_variables
                raise ValueError(
                    f"Variables in relationships not found in all_variables: {missing}"
                )
        else:
            self.all_variables = self._extract_variables_from_relationships()
    
    def _validate_and_add_relationship(self, item: Tuple[str, str, str]) -> None:
        """
        Validate and add a relationship tuple.
        
        Args:
            item: Tuple of (var1, var2, relationship)
        
        Raises:
            TypeError: If item is not a tuple or has wrong length
            ValueError: If relationship is not valid
        """
        # Check if item is a tuple-like structure
        try:
            if len(item) != 3:
                raise ValueError(
                    f"Relationship tuple must have exactly 3 elements (var1, var2, relationship), "
                    f"got {len(item)} elements: {item}"
                )
        except TypeError:
            raise TypeError(
                f"Each relationship must be a tuple/iterable, got {type(item)}: {item}"
            )
        
        var1, var2, relationship = item
        
        # Validate types
        if not isinstance(var1, str) or not isinstance(var2, str):
            raise TypeError(
                f"Variable names must be strings, got var1={type(var1)}, var2={type(var2)}"
            )
        
        if not isinstance(relationship, str):
            raise TypeError(
                f"Relationship must be a string, got {type(relationship)}"
            )
        
        # Validate relationship type
        if relationship not in self.VALID_RELATIONSHIPS:
            raise ValueError(
                f"Invalid relationship '{relationship}'. Must be one of: "
                f"{', '.join(sorted(self.VALID_RELATIONSHIPS))}"
            )
        
        # Add to set
        relationship_tuple = (var1, var2, relationship)
        self.relationships.add(relationship_tuple)
        self._relationship_dict[(var1, var2)] = relationship
    
    def _extract_variables_from_relationships(self) -> Set[str]:
        """
        Extract all unique variables from the relationships.
        
        Returns:
            Set of all variable names
        """
        variables = set()
        for var1, var2, _ in self.relationships:
            variables.add(var1)
            variables.add(var2)
        return variables
    
    def get_relationship(self, var1: str, var2: str) -> Optional[str]:
        """
        Get the relationship between two variables.
        
        Args:
            var1: First variable
            var2: Second variable
        
        Returns:
            Relationship string if exists, None otherwise
        """
        return self._relationship_dict.get((var1, var2))
    
    @classmethod
    def load_csv(cls, filename: str) -> 'SimpleCausalBenchmark':
        """
        Load a SimpleCausalBenchmark from a CSV file.
        
        Args:
            filename: Path to the CSV file
        
        Returns:
            SimpleCausalBenchmark instance loaded from the file
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the CSV format is invalid or data cannot be loaded
        """
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filename}")
        
        relationships = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # Skip the header row (first row)
                try:
                    next(reader)
                except StopIteration:
                    # Empty file
                    raise ValueError(f"CSV file is empty: {filename}")
                
                # Read data rows
                for row_num, row in enumerate(reader, start=2):  # Start at 2 since row 1 is header
                    # Check column count
                    if len(row) != 3:
                        raise ValueError(
                            f"Row {row_num} has {len(row)} columns, expected 3.\n"
                            f"Row data: {row}"
                        )
                    
                    # Convert all elements to strings and strip whitespace
                    var1 = str(row[0]).strip()
                    var2 = str(row[1]).strip()
                    relationship = str(row[2]).strip()
                    
                    # Validate that we have non-empty values
                    if not var1 or not var2 or not relationship:
                        raise ValueError(
                            f"Row {row_num} contains empty values.\n"
                            f"Row data: var1='{row[0]}', var2='{row[1]}', relationship='{row[2]}'"
                        )
                    
                    try:
                        # Try to add the relationship to validate it
                        relationships.append((var1, var2, relationship))
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load row {row_num}: {str(e)}\n"
                            f"Row data: var1='{var1}', var2='{var2}', relationship='{relationship}'"
                        ) from e
        
        except csv.Error as e:
            raise ValueError(f"CSV parsing error in {filename}: {str(e)}") from e
        
        # Create and return the benchmark
        try:
            return cls(relationships)
        except Exception as e:
            raise ValueError(f"Failed to create benchmark from CSV data: {str(e)}") from e
    
    def save_to_csv(self, filename: str) -> None:
        """
        Save this benchmark to a CSV file.
        
        Args:
            filename: Path where the CSV file should be saved
        
        Raises:
            IOError: If the file cannot be written
        """
        filepath = Path(filename)
        
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['var1', 'var2', 'relationship'])
                
                # Write data rows (sorted for consistency)
                for var1, var2, relationship in sorted(self.relationships):
                    writer.writerow([str(var1), str(var2), str(relationship)])
                    
        except Exception as e:
            raise IOError(f"Failed to save benchmark to {filename}: {str(e)}") from e
    
    def evaluate(self, other: 'SimpleCausalBenchmark') -> float:
        """
        Evaluate similarity between this benchmark (baseline) and another benchmark.
        
        The evaluation computes the proportion of matching relationships.
        For each relationship in the baseline, we check if it exists in the other benchmark.
        
        Args:
            other: Another SimpleCausalBenchmark to compare against
        
        Returns:
            Float similarity score in range [0.0, 1.0], where:
            - 1.0 means perfect match (all baseline relationships match)
            - 0.0 means no matches
        
        Raises:
            TypeError: If other is not a SimpleCausalBenchmark instance
        """
        if not isinstance(other, SimpleCausalBenchmark):
            raise TypeError(
                f"Can only evaluate against another SimpleCausalBenchmark, "
                f"got {type(other)}"
            )
        
        # If baseline is empty, return 1.0 (trivially perfect)
        if len(self.relationships) == 0:
            return 1.0
        
        # Count matching relationships
        matching = 0
        for relationship in self.relationships:
            if relationship in other.relationships:
                matching += 1
        
        # Calculate similarity as proportion of baseline relationships that match
        similarity = matching / len(self.relationships)
        return similarity
    
    def evaluate_detailed(self, other: 'SimpleCausalBenchmark') -> Dict[str, float]:
        """
        Evaluate similarity with detailed metrics.
        
        Args:
            other: Another SimpleCausalBenchmark to compare against
        
        Returns:
            Dictionary containing:
            - 'accuracy': Overall accuracy
            - 'precision': Precision (of other's predictions)
            - 'recall': Recall (how many baseline relationships were found)
            - 'f1': F1 score
            - 'total_baseline': Number of relationships in baseline
            - 'total_other': Number of relationships in other
            - 'matching': Number of matching relationships
        """
        if not isinstance(other, SimpleCausalBenchmark):
            raise TypeError(
                f"Can only evaluate against another SimpleCausalBenchmark, "
                f"got {type(other)}"
            )
        
        baseline_count = len(self.relationships)
        other_count = len(other.relationships)
        matching = len(self.relationships & other.relationships)
        
        # Calculate metrics
        recall = matching / baseline_count if baseline_count > 0 else 1.0
        precision = matching / other_count if other_count > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy: matching / total unique relationships
        all_relationships = self.relationships | other.relationships
        accuracy = matching / len(all_relationships) if len(all_relationships) > 0 else 1.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_baseline': baseline_count,
            'total_other': other_count,
            'matching': matching
        }
    
    def __len__(self) -> int:
        """Return number of relationships."""
        return len(self.relationships)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SimpleCausalBenchmark("
            f"relationships={len(self.relationships)}, "
            f"variables={len(self.all_variables)})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"SimpleCausalBenchmark with {len(self.relationships)} relationships:"]
        for var1, var2, rel in sorted(self.relationships):
            lines.append(f"  {var1} {rel} {var2}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("SimpleCausalBenchmark Example Usage")
    print("=" * 60)
    
    # Create a benchmark
    relationships = [
        ('A', 'B', '->'),
        ('B', 'C', '->'),
        ('C', 'A', 'independent'),
        ('D', 'E', '<->'),
    ]
    
    print("\n1. Creating benchmark from relationships:")
    benchmark1 = SimpleCausalBenchmark(relationships)
    print(benchmark1)
    print(f"All variables: {sorted(benchmark1.all_variables)}")
    
    # Create another benchmark for comparison
    print("\n2. Creating second benchmark:")
    relationships2 = [
        ('A', 'B', '->'),      # Match
        ('B', 'C', '<-'),      # Different
        ('C', 'A', 'independent'),  # Match
        ('D', 'E', '->'),      # Different
    ]
    benchmark2 = SimpleCausalBenchmark(relationships2)
    print(benchmark2)
    
    # Evaluate
    print("\n3. Evaluation Results:")
    similarity = benchmark1.evaluate(benchmark2)
    print(f"Simple similarity: {similarity:.2%}")
    
    detailed = benchmark1.evaluate_detailed(benchmark2)
    print("\nDetailed metrics:")
    for key, value in detailed.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    # Test CSV save and load
    print("\n4. Testing CSV save and load:")
    test_file = "test_benchmark_temp.csv"
    
    try:
        # Save to CSV
        print(f"  Saving benchmark to {test_file}...")
        benchmark1.save_to_csv(test_file)
        print(f"  ✓ Saved successfully")
        
        # Load from CSV
        print(f"  Loading benchmark from {test_file}...")
        loaded_benchmark = SimpleCausalBenchmark.load_csv(test_file)
        print(f"  ✓ Loaded successfully")
        
        # Verify they match
        if loaded_benchmark.relationships == benchmark1.relationships:
            print("  ✓ Loaded benchmark matches original")
        else:
            print("  ✗ Loaded benchmark does not match original")
        
        # Display loaded content
        print(f"\n  Loaded benchmark:")
        print(f"    Relationships: {len(loaded_benchmark.relationships)}")
        print(f"    Variables: {sorted(loaded_benchmark.all_variables)}")
        
    finally:
        # Clean up test file
        import os
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n  Cleaned up test file: {test_file}")
    
    # Test error handling
    print("\n5. Testing error handling:")
    
    # Test invalid relationship
    try:
        bad_benchmark = SimpleCausalBenchmark([('A', 'B', 'invalid_rel')])
    except ValueError as e:
        print(f"✓ Caught expected error for invalid relationship")
    
    # Test wrong tuple length
    try:
        bad_benchmark = SimpleCausalBenchmark([('A', 'B')])  # Missing relationship
    except ValueError as e:
        print(f"✓ Caught expected error for wrong tuple length")
    
    # Test loading non-existent file
    try:
        bad_benchmark = SimpleCausalBenchmark.load_csv("nonexistent_file.csv")
    except FileNotFoundError as e:
        print(f"✓ Caught expected error for non-existent file")
    
    # Test loading invalid CSV (wrong number of columns)
    print("\n6. Testing CSV error handling:")
    bad_csv_file = "test_bad_csv_temp.csv"
    try:
        # Create a CSV with wrong number of columns
        with open(bad_csv_file, 'w') as f:
            f.write("var1,var2,relationship\n")
            f.write("A,B\n")  # Only 2 columns
        
        try:
            bad_benchmark = SimpleCausalBenchmark.load_csv(bad_csv_file)
        except ValueError as e:
            print(f"✓ Caught expected error for wrong column count:")
            print(f"  {str(e)[:100]}...")
    finally:
        if os.path.exists(bad_csv_file):
            os.remove(bad_csv_file)
    
    # Test loading CSV with invalid relationship
    bad_rel_file = "test_bad_rel_temp.csv"
    try:
        with open(bad_rel_file, 'w') as f:
            f.write("var1,var2,relationship\n")
            f.write("A,B,invalid_relationship\n")
        
        try:
            bad_benchmark = SimpleCausalBenchmark.load_csv(bad_rel_file)
        except ValueError as e:
            print(f"✓ Caught expected error for invalid relationship in CSV:")
            print(f"  {str(e)[:100]}...")
    finally:
        if os.path.exists(bad_rel_file):
            os.remove(bad_rel_file)
    
    # Test loading CSV with empty values
    empty_val_file = "test_empty_val_temp.csv"
    try:
        with open(empty_val_file, 'w') as f:
            f.write("var1,var2,relationship\n")
            f.write("A,,->\n")  # Empty var2
        
        try:
            bad_benchmark = SimpleCausalBenchmark.load_csv(empty_val_file)
        except ValueError as e:
            print(f"✓ Caught expected error for empty value in CSV:")
            print(f"  {str(e)[:100]}...")
    finally:
        if os.path.exists(empty_val_file):
            os.remove(empty_val_file)
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
