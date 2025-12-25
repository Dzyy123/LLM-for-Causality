"""
Test Script for Causal Benchmark - Multiprocessing Version

This script runs ALL benchmark CSV files in the /benchmarks folder simultaneously
using multiprocessing, with a rich progress bar showing overall progress.

Usage:
    python TEST_causal_benchmark.py

Configuration:
    - Edit the GLOBAL CONFIGURATION section below to change settings
    - BENCHMARKS_DIR: Directory containing benchmark CSV files
    - OUTPUT_DIR: Directory where results will be saved
    - NUM_PROCESSES: Number of parallel processes (None = use CPU count)

Output:
    - Results saved to: {OUTPUT_DIR}/{benchmark_name}_results.csv
    - Yes/No data saved to: {OUTPUT_DIR}/{benchmark_name}_yes_no.csv
    
Features:
    - Runs all benchmarks in parallel using multiprocessing
    - Rich progress bar shows completion status
    - Each benchmark runs in its own process
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
from queue import Empty

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from causal_benchmark import SimpleCausalBenchmark
from causal_benchmark_test_all import BenchmarkRunner
from llm_utils import OnlineLLMClient, setup_logging, get_report_logger
from tree_query import CausalDiscoveryFramework

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Benchmarks directory (contains all CSV files to process)
BENCHMARKS_DIR = "benchmarks"

# Output directory for results
OUTPUT_DIR = "results"

# Multiprocessing configuration
NUM_PROCESSES = None  # None = use mp.cpu_count()

# Framework Configuration
M_SAMPLES = 5  # Number of original answer samples
TRUST_CONFIDENCE = 0.5  # Confidence threshold
SEED = 42
MAX_WORKERS = 4  # Number of parallel threads for processing variable pairs (increase for speed)

# Cache Configuration
CLEAR_EXPERT_CACHE = False  # Set to True to rebuild expert cache from scratch

# Report Logging Configuration
ENABLE_REPORT_LOGGING = True  # Set to False to disable LLM interaction logging

# LLM Configuration
API_KEY = "sk-ywkSCgAfmNoHn2R0rKeGUg"
BASE_URL = "https://llmapi.paratera.com/v1/"
MODEL_NAME = "Qwen3-Next-80B-A3B-Instruct"
MAX_TOKENS = 800
TEMPERATURE = 0.7

# Logging Configuration
WORKER_LOG_LEVEL = "INFO"  # Set to INFO to see detailed progress, WARNING to only see errors
MAX_LOG_LINES = 20  # Maximum number of log lines to display in Live view

# ============================================================================
# QUEUE-BASED LOGGING HANDLER
# ============================================================================


class QueueHandler(logging.Handler):
    """Custom handler that sends log records to a multiprocessing Queue."""
    
    def __init__(self, log_queue: mp.Queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(('log', record.name, record.levelname, msg))
        except Exception:
            self.handleError(record)

# ============================================================================
# WORKER FUNCTION (runs in separate process)
# ============================================================================


def run_single_benchmark(benchmark_file: Path, log_queue: mp.Queue) -> Tuple[str, bool, str]:
    """
    Run a single benchmark file in a separate process.
    
    Args:
        benchmark_file: Path to the benchmark CSV file
        log_queue: Queue for sending log messages back to main process
        
    Returns:
        Tuple of (benchmark_name, success, message)
    """
    benchmark_name = benchmark_file.stem
    
    try:
        # Setup logging for this process
        # Use WARNING level and QueueHandler to send logs to main process
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(getattr(logging, WORKER_LOG_LEVEL))
        
        # Add queue handler to send important logs to main process
        queue_handler = QueueHandler(log_queue)
        queue_handler.setLevel(logging.INFO)  # Send INFO and above to main process
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', 
                                     datefmt='%m-%d %H:%M:%S')
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)
        
        logger = logging.getLogger(f"worker.{benchmark_name}")
        logger.info(f"Starting benchmark: {benchmark_name}")
        
        # Initialize report logger (disable console output)
        report_logger = get_report_logger()
        report_logger.enabled = ENABLE_REPORT_LOGGING
        
        # Load benchmark
        benchmark = SimpleCausalBenchmark.load_csv(str(benchmark_file))
        
        # Initialize LLM client
        client = OnlineLLMClient(
            api_key=API_KEY,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        # Start report if logging is enabled
        if ENABLE_REPORT_LOGGING:
            config_info = {
                "Benchmark File": str(benchmark_file),
                "Model": MODEL_NAME,
                "Base URL": BASE_URL,
                "Temperature": TEMPERATURE,
                "Max Tokens": MAX_TOKENS,
                "M Samples": M_SAMPLES,
                "Trust Confidence": TRUST_CONFIDENCE,
            }
            report_logger.start_new_report(
                report_name=benchmark_name,
                config_info=config_info
            )
        
        # Create causal discovery framework
        framework = CausalDiscoveryFramework(
            client=client,
            all_variables=list(benchmark.all_variables),
            m_samples=M_SAMPLES,
            trust_confidence=TRUST_CONFIDENCE,
            seed=SEED,
            max_workers=MAX_WORKERS
        )
        
        # Create benchmark runner
        runner = BenchmarkRunner(
            framework=framework,
            output_dir=OUTPUT_DIR,
            seed=SEED
        )
        
        # Run benchmark
        output_filename = f"{benchmark_name}_results.csv"
        output_path = runner.run_on_benchmark(
            benchmark=benchmark,
            output_filename=output_filename,
            include_metadata=True
        )
        
        # Export Yes/No results
        yes_no_filename = f"{benchmark_name}_yes_no.csv"
        yes_no_path = runner.export_yes_no_results(yes_no_filename)
        
        # Close report if logging is enabled
        if ENABLE_REPORT_LOGGING:
            summary = (
                f"Benchmark test completed.\n"
                f"Variables: {len(benchmark.all_variables)}\n"
                f"Results: {output_path}\n"
                f"Yes/No: {yes_no_path}\n"
                f"Records: {len(runner.yes_no_data)}"
            )
            report_logger.close_report(summary=summary)
        
        logger.info(f"Completed benchmark: {benchmark_name}")
        message = f"✓ {output_filename}, {yes_no_filename}"
        log_queue.put(('complete', benchmark_name, True, message))
        return (benchmark_name, True, message)
        
    except Exception as e:
        logger = logging.getLogger(f"worker.{benchmark_name}")
        logger.error(f"Failed benchmark {benchmark_name}: {e}", exc_info=True)
        message = f"✗ Error: {str(e)}"
        log_queue.put(('complete', benchmark_name, False, message))
        return (benchmark_name, False, message)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """
    Main execution function.
    
    Steps:
    1. Discover all CSV files in benchmarks directory
    2. Clear expert cache if requested
    3. Run all benchmarks in parallel with progress bar
    4. Report results
    """
    # Setup logging for main process with RichHandler
    setup_logging(level="INFO")  # Disable rich handler to avoid conflicts
    logger = logging.getLogger(__name__)
    console = Console()
    
    console.print("[bold cyan]TEST_causal_benchmark.py - Multiprocessing Mode")
    console.print("[bold cyan]=" * 70 + "\n")
    
    # Step 1: Discover all CSV benchmark files
    console.print("[bold]Step 1:[/bold] Discovering benchmark files...")
    benchmarks_path = Path(BENCHMARKS_DIR)
    
    if not benchmarks_path.exists():
        console.print(f"[bold red]✗ Benchmarks directory not found: {BENCHMARKS_DIR}[/bold red]")
        return 1
    
    # Find all CSV files
    csv_files = sorted(benchmarks_path.glob("*.csv"))
    # Filter out non-benchmark files (e.g., files not starting with "benchmark_")
    benchmark_files = [f for f in csv_files if f.name.startswith("benchmark_")]
    
    if not benchmark_files:
        console.print(f"[bold red]✗ No benchmark CSV files found in {BENCHMARKS_DIR}[/bold red]")
        return 1
    
    console.print(f"[green]✓ Found {len(benchmark_files)} benchmark files:[/green]")
    for bf in benchmark_files:
        console.print(f"  • {bf.name}")
    
    # Step 2: Clear expert cache if requested
    if CLEAR_EXPERT_CACHE:
        console.print("\n[bold]Step 2:[/bold] Clearing expert cache...")
        cache_dir = Path(".expert_cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            console.print(f"[green]✓ Removed cache directory: {cache_dir}[/green]")
    else:
        console.print("\n[bold]Step 2:[/bold] Skipping cache clear (CLEAR_EXPERT_CACHE=False)")
    
    # Step 3: Run all benchmarks in parallel
    console.print(f"\n[bold]Step 3:[/bold] Running {len(benchmark_files)} benchmarks in parallel...")
    
    # Determine number of processes
    num_procs = NUM_PROCESSES if NUM_PROCESSES else mp.cpu_count()
    console.print(f"[dim]Using {num_procs} parallel processes[/dim]\n")
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Create a queue for log messages
    log_queue = mp.Manager().Queue()
    
    # Store recent log messages
    recent_logs = []
    results = []
    
    # Create progress bar with detailed information
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),  # Shows [finished/total]
        TaskProgressColumn(),  # Shows percentage
        TextColumn("•"),
        TimeElapsedColumn(),  # Shows elapsed time
        TextColumn("•"),
        TimeRemainingColumn(),  # Shows estimated remaining time
        console=console
    )
    
    task_id = progress.add_task(
        "Benchmark Processing",  # Clearer description
        total=len(benchmark_files)
    )
    
    # Create log panel
    def make_log_panel():
        if not recent_logs:
            return Panel("Waiting for logs...", title="Recent Logs", border_style="blue")
        
        log_text = Text()
        for log_level, log_msg in recent_logs[-MAX_LOG_LINES:]:
            # Color code by log level
            if log_level == "ERROR":
                log_text.append(log_msg + "\n", style="bold red")
            elif log_level == "WARNING":
                log_text.append(log_msg + "\n", style="yellow")
            elif log_level == "INFO":
                log_text.append(log_msg + "\n", style="green")
            else:
                log_text.append(log_msg + "\n")
        
        return Panel(log_text, title=f"Recent Logs (last {MAX_LOG_LINES})", border_style="blue")
    
    # Use Live to manage both progress and logs
    with Live(Group(progress, make_log_panel()), console=console, refresh_per_second=4) as live:
        # Use multiprocessing pool
        with mp.Pool(processes=num_procs) as pool:
            # Submit all tasks
            async_results = [
                pool.apply_async(run_single_benchmark, (bf, log_queue))
                for bf in benchmark_files
            ]
            
            # Collect results as they complete
            completed = 0
            while completed < len(benchmark_files):
                # Check for log messages
                try:
                    while True:
                        msg = log_queue.get_nowait()
                        if msg[0] == 'log':
                            _, logger_name, level, log_msg = msg
                            recent_logs.append((level, log_msg))
                            # Update display
                            live.update(Group(progress, make_log_panel()))
                        elif msg[0] == 'complete':
                            _, name, success, message = msg
                            # Update progress
                            progress.advance(task_id)
                            completed += 1
                except Empty:
                    pass
                
                # Check if any async result is ready
                for async_result in async_results:
                    if async_result.ready() and async_result not in [r[0] for r in results]:
                        result = async_result.get()
                        results.append((async_result, result))
                
                # Small sleep to avoid busy waiting
                import time
                time.sleep(0.1)
    
    # Extract final results
    final_results = [r[1] for r in results]
    
    # Step 4: Report results
    console.print(f"\n[bold]Step 4:[/bold] Summary of results")
    console.print("[cyan]" + "=" * 70 + "[/cyan]")
    
    success_count = sum(1 for _, success, _ in final_results if success)
    failure_count = len(final_results) - success_count
    
    for name, success, message in final_results:
        if success:
            console.print(f"[green]{name}:[/green] {message}")
        else:
            console.print(f"[red]{name}:[/red] {message}")
    
    console.print("[cyan]" + "=" * 70 + "[/cyan]")
    console.print(f"\n[bold]Total:[/bold] {len(final_results)} benchmarks")
    console.print(f"[bold green]Success:[/bold green] {success_count}")
    console.print(f"[bold red]Failed:[/bold red] {failure_count}")
    
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("TEST_causal_benchmark.py - Completed")
    console.print("=" * 70 + "[/bold cyan]\n")
    
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    exit(main())
