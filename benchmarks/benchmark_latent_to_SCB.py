"""
Convert benchmark_latent.xlsx to separate CSV files grouped by dataset.
Transforms ground truth format to standardized symbolic format.
"""
import pandas as pd
from pathlib import Path

# Minimum number of variable pairs required to output a dataset
DATA_NUM_THRESHOLD = 5


def convert_relationship(rel: str, var1: str, var2: str) -> str:
    """
    Convert relationship format to standardized symbolic format.
    
    Args:
        rel: Original relationship string (e.g., '->', '<-', '<->', or other)
        var1: First variable name (not used in output, kept for compatibility)
        var2: Second variable name (not used in output, kept for compatibility)
        
    Returns:
        Standardized relationship string: '->', '<-', '<->', or 'independent'
    """
    rel = rel.strip()
    if rel == '->':
        return "->"
    elif rel == '<-':
        return "<-"
    elif rel == '<->':
        return "<->"
    else:
        # Treat all other cases as independent
        return "independent"


def main():
    # Define paths
    input_file = "benchmark_latent.xlsx"
    output_dir = Path("benchmarks") 
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Read the Excel file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    # Display basic info
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Group by dataset
    # Try to find dataset column with different case or name
    dataset_cols = [col for col in df.columns if 'dataset' in col.lower()]
    if dataset_cols:
        dataset_col = dataset_cols[0]
        print(f"Using column '{dataset_col}' as dataset column")
    else:
        print("Error: No 'dataset' column found in the Excel file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Get unique datasets
    datasets = df[dataset_col].unique()
    print(f"\nFound {len(datasets)} datasets: {list(datasets)}")
    
    # Process each dataset
    for dataset_name in datasets:
        # Filter rows for this dataset
        dataset_df = df[df[dataset_col] == dataset_name].copy()
        
        # Check if dataset meets minimum threshold
        if len(dataset_df) < DATA_NUM_THRESHOLD:
            print(f"  Skipping '{dataset_name}' (only {len(dataset_df)} pairs, threshold is {DATA_NUM_THRESHOLD})")
            continue
        
        # Clean dataset name for use in filename (remove invalid characters)
        safe_dataset_name = dataset_name.replace('"', '').replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        
        # Identify variable and relationship columns
        # Common column names: var1, var2, variable1, variable2, relationship, ground_truth, etc.
        var1_col = None
        var2_col = None
        rel_col = None
        
        for col in dataset_df.columns:
            col_lower = col.lower().replace(' ', '_')
            col_original_lower = col.lower()
            if col_lower in ['var1', 'variable1', 'v1']:
                var1_col = col
            elif col_lower in ['var2', 'variable2', 'v2']:
                var2_col = col
            elif col_lower in ['relationship', 'ground_truth', 'gt', 'relation'] or col_original_lower in ['ground truth']:
                rel_col = col
        
        if not var1_col or not var2_col or not rel_col:
            print(f"Warning: Could not identify all required columns for dataset '{dataset_name}'")
            print(f"  var1: {var1_col}, var2: {var2_col}, relationship: {rel_col}")
            continue
        
        # Convert relationship format
        dataset_df['relationship'] = dataset_df.apply(
            lambda row: convert_relationship(row[rel_col], row[var1_col], row[var2_col]),
            axis=1
        )
        
        # Create output dataframe with standardized column names
        output_df = pd.DataFrame({
            'var1': dataset_df[var1_col],
            'var2': dataset_df[var2_col],
            'relationship': dataset_df['relationship']
        })
        
        # Save to CSV
        output_file = output_dir / f"benchmark_latent_{safe_dataset_name}.csv"
        output_df.to_csv(output_file, index=False)
        print(f"  Saved {len(output_df)} rows to {output_file.name}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
