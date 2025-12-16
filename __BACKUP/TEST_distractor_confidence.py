"""
Test Distractor Confidence Estimator

Uses logging for progress tracking and rich formatting for output.
"""

import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

from llm_utils import OnlineLLMClient, setup_logging
from tree_query import (
    BackdoorPathExpert,
    IndependenceExpert,
    CausalDirectionExpert,
    LatentConfounderExpert,
    create_distractor_confidence_estimator
)
from distractor_confidence_estimator_2 import create_distractor_confidence_estimator_v2


# Initialize rich console
console = Console()


class MarkdownReportWriter:
    """Writes test results in Markdown format."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
    
    def write_header(self):
        """Write report header."""
        self.file.write("# Distractor Confidence Estimator Test Report\n\n")
        self.file.write(f"**Test Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file.write("---\n\n")
    
    def write_config(self, client, estimator):
        """Write configuration section."""
        self.file.write("## Configuration\n\n")
        self.file.write("### LLM Client\n\n")
        self.file.write(f"- **API Base:** {client.base_url}\n")
        self.file.write(f"- **Model:** {client.model_name}\n")
        self.file.write(f"- **Max Tokens:** {client.max_tokens}\n")
        self.file.write(f"- **Temperature:** {client.temperature}\n\n")
        
        self.file.write("### Estimator Parameters\n\n")
        self.file.write(f"- **K1 Samples:** {estimator.k1_samples}\n")
        self.file.write(f"- **K2 Samples:** {estimator.k2_samples}\n")
        self.file.write(f"- **Total Samples:** {estimator.k1_samples * estimator.k2_samples * 3}\n")
        self.file.write(f"- **Seed:** {estimator.initial_seed if estimator.initial_seed else 'Random'}\n\n")
        self.file.write("---\n\n")
        self.flush()
    
    def write_test_header(self, test_num: int, test_name: str, description: str):
        """Write test section header."""
        self.file.write(f"## Test {test_num}: {test_name}\n\n")
        self.file.write(f"**Description:** {description}\n\n")
        self.flush()
    
    def write_expert_config(self, expert, all_variables: List[str]):
        """Write expert configuration."""
        self.file.write("### Expert Configuration\n\n")
        self.file.write(f"- **Type:** {expert.expert_type}\n")
        self.file.write(f"- **Variable X1:** {expert.x1}\n")
        self.file.write(f"- **Variable X2:** {expert.x2}\n")
        self.file.write(f"- **All Variables:** {', '.join(all_variables)}\n\n")
        
        self.file.write("**Base Question:**\n\n")
        self.file.write(f"```\n{expert.base_prompt}\n```\n\n")
        self.flush()
    
    def write_original_answers(self, original_answers):
        """Write original answers section."""
        self.file.write("### Original Answers\n\n")
        self.file.write(f"Total: {len(original_answers)} answers\n\n")
        
        for answer in original_answers:
            result = "Yes" if answer.label == 1 else "No"
            self.file.write(f"#### Answer {answer.index + 1} (Result: {result})\n\n")
            self.file.write(f"```\n{answer.text}\n```\n\n")
        
        self.flush()
    
    def write_distractor_sets(self, distractor_sets, max_sets: int = 2):
        """Write distractor sets (limited to first few)."""
        self.file.write("### Distractor Sets\n\n")
        self.file.write(f"Total: {len(distractor_sets)} sets (showing first {max_sets})\n\n")
        
        for dset in distractor_sets[:max_sets]:
            orig_result = "Yes" if dset.original_label == 1 else "No"
            self.file.write(f"#### Set {dset.set_index + 1} (Based on: {orig_result})\n\n")
            
            for dtype in ['contrarian', 'deceiver', 'hater']:
                distractor = getattr(dset, dtype)
                self.file.write(f"**{dtype.upper()}:**\n\n")
                self.file.write(f"<details>\n<summary>Generation Prompt</summary>\n\n")
                self.file.write(f"```\n{distractor.prompt}\n```\n\n")
                self.file.write(f"</details>\n\n")
                self.file.write(f"**Generated Argument:**\n\n")
                self.file.write(f"```\n{distractor.argument}\n```\n\n")
        
        self.flush()
    
    def write_distracted_samples(self, samples, max_samples: int = 3):
        """Write distracted samples (limited to first few per type)."""
        self.file.write("### Distracted Sample Results\n\n")
        
        # Group by distractor type
        samples_by_type = {'contrarian': [], 'deceiver': [], 'hater': []}
        for sample in samples:
            samples_by_type[sample.distractor_type].append(sample)
        
        for dtype in ['contrarian', 'deceiver', 'hater']:
            type_samples = samples_by_type[dtype]
            if not type_samples:
                continue
            
            self.file.write(f"#### {dtype.upper()} Distractor\n\n")
            self.file.write(f"Total samples: {len(type_samples)}\n\n")
            
            for i, sample in enumerate(type_samples[:max_samples]):
                result = "Yes" if sample.label == 1 else "No"
                orig = "Yes" if sample.original_label == 1 else "No"
                flipped = " (FLIPPED)" if sample.label != sample.original_label else ""
                
                self.file.write(f"**Sample {i+1}:** {result}{flipped} (Original: {orig})\n\n")
                self.file.write(f"<details>\n<summary>Sampling Prompt</summary>\n\n")
                self.file.write(f"```\n{sample.sampling_prompt}\n```\n\n")
                self.file.write(f"</details>\n\n")
                self.file.write(f"**Response:**\n\n")
                self.file.write(f"```\n{sample.response}\n```\n\n")
        
        self.flush()
    
    def write_detailed_tracking(self, metrics: Dict[str, Any], k1_samples: int, k2_samples: int):
        """Write detailed tracking of each answer through distractors."""
        self.file.write("### Detailed Answer Tracking\n\n")
        self.file.write("Shows how each distractor set changed the answer. ")
        self.file.write(f"Each of {k1_samples} original answers generates {k2_samples} distractor set(s).\n\n")
        
        tracking = metrics['detailed_tracking']
        if not tracking:
            self.file.write("No tracking data available.\n\n")
            return
        
        self.file.write("| Orig# | Set# | Original | Contrarian | Deceiver | Hater | Flips |\n")
        self.file.write("|-------|------|----------|------------|----------|-------|-------|\n")
        
        for set_idx in sorted(tracking.keys()):
            data = tracking[set_idx]
            orig_idx = (set_idx // k2_samples) + 1  # Which original answer
            set_num = (set_idx % k2_samples) + 1     # Which distractor set for that answer
            
            orig = "Yes" if data['original_label'] == 1 else "No"
            contr = "Yes" if data['contrarian'] == 1 else "No"
            decv = "Yes" if data['deceiver'] == 1 else "No"
            hatr = "Yes" if data['hater'] == 1 else "No"
            
            # Count flips
            flips = 0
            if data['contrarian'] != data['original_label']: flips += 1
            if data['deceiver'] != data['original_label']: flips += 1
            if data['hater'] != data['original_label']: flips += 1
            
            # Highlight if flipped
            contr_str = f"**{contr}**" if data['contrarian'] != data['original_label'] else contr
            decv_str = f"**{decv}**" if data['deceiver'] != data['original_label'] else decv
            hatr_str = f"**{hatr}**" if data['hater'] != data['original_label'] else hatr
            
            self.file.write(f"| {orig_idx} | {set_num} | {orig} | {contr_str} | {decv_str} | {hatr_str} | {flips}/3 |\n")
        
        self.file.write("\n*Note: Orig# = Original answer number, Set# = Distractor set number for that answer. Bold values indicate flipped answers.*\n\n")
        self.flush()
    
    def write_metrics_table(self, expert_result: str, metrics: Dict[str, Any]):
        """Write metrics as a table."""
        self.file.write("### Summary Statistics\n\n")
        self.file.write(f"**Expert Judgment:** {expert_result}\n\n")
        
        self.file.write("#### Results by Distractor Type\n\n")
        self.file.write("| Distractor Type | Yes Count | No Count | Flip Rate | Agreement Rate |\n")
        self.file.write("|----------------|-----------|----------|-----------|----------------|\n")
        for dtype, display_name in [('contrarian', 'Contrarian'), ('deceiver', 'Deceiver'), ('hater', 'Hater')]:
            dist = metrics['label_distributions'][dtype]
            flip_rate = metrics['flip_rates'][dtype]
            agreement = metrics[f"p{['contrarian', 'deceiver', 'hater'].index(dtype) + 1}"]
            self.file.write(f"| {display_name} | {dist['yes']} | {dist['no']} | {flip_rate:.2%} | {agreement:.2%} |\n")
        self.file.write(f"| **Average** | - | - | **{metrics['avg_flip_rate']:.2%}** | - |\n\n")
        
        self.file.write("#### Key Metrics\n\n")
        self.file.write("| Metric | Value |\n")
        self.file.write("|--------|-------|\n")
        self.file.write(f"| Confidence Score | {metrics['confidence_score']:.4f} |\n")
        self.file.write(f"| Robustness Score | {metrics['robustness_score']:.2%} |\n")
        self.file.write(f"| K1 Samples | {metrics['k1_samples']} |\n")
        self.file.write(f"| K2 Samples | {metrics['k2_samples']} |\n")
        self.file.write(f"| Total Samples | {metrics['total_samples']} |\n\n")
        
        self.file.write("---\n\n")
        self.flush()
    
    def flush(self):
        """Flush file buffer."""
        self.file.flush()
    
    def close(self):
        """Close file."""
        self.file.write("\n---\n\n")
        self.file.write("*End of Report*\n")
        self.file.close()


def display_detailed_tracking(metrics: Dict[str, Any], k1_samples: int, k2_samples: int):
    """Display detailed tracking of each answer through distractors."""
    tracking = metrics.get('detailed_tracking', {})
    if not tracking:
        return
    
    table = Table(title=f"Detailed Answer Tracking ({k1_samples} answers × {k2_samples} sets = {k1_samples * k2_samples} total)", box=box.ROUNDED)
    table.add_column("Orig#", style="cyan", justify="center")
    table.add_column("Set#", style="magenta", justify="center")
    table.add_column("Original", style="white", justify="center")
    table.add_column("Contrarian", justify="center")
    table.add_column("Deceiver", justify="center")
    table.add_column("Hater", justify="center")
    table.add_column("Flips", style="yellow", justify="center")
    
    for set_idx in sorted(tracking.keys()):
        data = tracking[set_idx]
        orig_idx = (set_idx // k2_samples) + 1  # Which original answer
        set_num = (set_idx % k2_samples) + 1     # Which distractor set for that answer
        
        orig = "Yes" if data['original_label'] == 1 else "No"
        contr = "Yes" if data['contrarian'] == 1 else "No"
        decv = "Yes" if data['deceiver'] == 1 else "No"
        hatr = "Yes" if data['hater'] == 1 else "No"
        
        # Count flips
        flips = 0
        if data['contrarian'] != data['original_label']: flips += 1
        if data['deceiver'] != data['original_label']: flips += 1
        if data['hater'] != data['original_label']: flips += 1
        
        # Color code flipped answers
        contr_style = "red" if data['contrarian'] != data['original_label'] else "green"
        decv_style = "red" if data['deceiver'] != data['original_label'] else "green"
        hatr_style = "red" if data['hater'] != data['original_label'] else "green"
        
        table.add_row(
            str(orig_idx),
            str(set_num),
            orig,
            f"[{contr_style}]{contr}[/{contr_style}]",
            f"[{decv_style}]{decv}[/{decv_style}]",
            f"[{hatr_style}]{hatr}[/{hatr_style}]",
            f"{flips}/3"
        )
    
    console.print(table)
    console.print("[dim]Note: Orig# = Original answer, Set# = Distractor set. Green = No flip, Red = Flipped[/dim]\n")


def display_metrics_table(expert_result: str, metrics: Dict[str, Any]):
    """Display metrics table in console."""
    # Results table with Yes-No distributions
    table = Table(title="Summary Statistics", box=box.ROUNDED)
    table.add_column("Distractor", style="cyan")
    table.add_column("Yes Count", style="green", justify="right")
    table.add_column("No Count", style="red", justify="right")
    table.add_column("Flip Rate", style="magenta", justify="right")
    table.add_column("Agreement", style="yellow", justify="right")
    
    for dtype, display_name, p_key in [("contrarian", "Contrarian", "p1"), ("deceiver", "Deceiver", "p2"), ("hater", "Hater", "p3")]:
        dist = metrics['label_distributions'][dtype]
        flip_rate = metrics['flip_rates'][dtype]
        agreement = metrics[p_key]
        table.add_row(
            display_name,
            str(dist['yes']),
            str(dist['no']),
            f"{flip_rate:.2%}",
            f"{agreement:.2%}"
        )
    
    table.add_row("Average", "-", "-", f"{metrics['avg_flip_rate']:.2%}", "-", style="bold")
    
    console.print(table)
    
    # Key metrics table
    metrics_table = Table(title="Key Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="yellow")
    
    metrics_table.add_row("Expert Judgment", expert_result)
    metrics_table.add_row("Confidence Score", f"{metrics['confidence_score']:.4f}")
    metrics_table.add_row("Robustness Score", f"{metrics['robustness_score']:.2%}")
    
    console.print(metrics_table)


def run_test(test_num: int, test_name: str, description: str,
             expert_class, expert_args: Dict[str, Any],
             client, estimator_v1, estimator_v2, writer: MarkdownReportWriter):
    """Run a single test case with both estimator versions."""
    console.rule(f"[bold blue]Test {test_num}: {test_name}")
    console.print(f"[dim]{description}[/dim]\n")
    
    # Create expert
    expert = expert_class(**expert_args, client=client)
    expert.base_prompt = expert.generate_question()
    
    # Display expert configuration
    config_table = Table(title="Expert Configuration", box=box.SIMPLE, show_header=False)
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Expert Type", expert.expert_type)
    config_table.add_row("Variable X1", expert.x1)
    config_table.add_row("Variable X2", expert.x2)
    config_table.add_row("All Variables", ", ".join(expert_args.get('all_variables', [])))
    console.print(config_table)
    
    console.print(f"\n[bold]Base Question:[/bold]")
    console.print(Panel(expert.base_prompt, border_style="dim", padding=(0, 2)))
    
    # Write test header
    writer.write_test_header(test_num, test_name, description)
    writer.write_expert_config(expert, expert_args.get('all_variables', []))
    
    # Get expert judgment
    result = expert.judge()
    expert_result_str = "Yes" if result['label'] == 1 else "No"
    
    console.print(f"\n[bold]Expert Judgment:[/bold] {expert_result_str}\n")
    
    # Version 1: With Full Explanation (Main Version)
    console.print("[bold cyan]Version 1: Previous Answer (Yes/No + Full Explanation)[/bold cyan]\n")
    confidence_result_v1 = estimator_v1.estimate_confidence(expert, result['label'])
    
    # Display V1 metrics
    display_detailed_tracking(confidence_result_v1, confidence_result_v1['k1_samples'], confidence_result_v1['k2_samples'])
    display_metrics_table(expert_result_str, confidence_result_v1)
    
    # Version 2: Yes/No Only (Simplified)
    console.print("\n[bold magenta]Version 2: Previous Answer (Yes/No only)[/bold magenta]\n")
    confidence_result_v2 = estimator_v2.estimate_confidence(expert, result['label'])
    
    # Display V2 metrics
    display_detailed_tracking(confidence_result_v2, confidence_result_v2['k1_samples'], confidence_result_v2['k2_samples'])
    display_metrics_table(expert_result_str, confidence_result_v2)
    
    # Comparison Table
    console.print("\n[bold yellow]Version Comparison[/bold yellow]\n")
    comparison_table = Table(title="V1 vs V2 Comparison", box=box.ROUNDED)
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("V1 (Full)", style="blue", justify="right")
    comparison_table.add_column("V2 (Yes/No)", style="magenta", justify="right")
    comparison_table.add_column("Difference", style="yellow", justify="right")
    
    # Flip rates comparison
    for dtype in ['contrarian', 'deceiver', 'hater']:
        v1_flip = confidence_result_v1['flip_rates'][dtype]
        v2_flip = confidence_result_v2['flip_rates'][dtype]
        diff = v2_flip - v1_flip
        diff_str = f"{diff:+.2%}" if diff != 0 else "0.00%"
        comparison_table.add_row(
            f"{dtype.capitalize()} Flip Rate",
            f"{v1_flip:.2%}",
            f"{v2_flip:.2%}",
            diff_str
        )
    
    # Average flip rate
    v1_avg = confidence_result_v1['avg_flip_rate']
    v2_avg = confidence_result_v2['avg_flip_rate']
    comparison_table.add_row(
        "Average Flip Rate",
        f"{v1_avg:.2%}",
        f"{v2_avg:.2%}",
        f"{v2_avg - v1_avg:+.2%}",
        style="bold"
    )
    
    # Confidence scores
    v1_conf = confidence_result_v1['confidence_score']
    v2_conf = confidence_result_v2['confidence_score']
    comparison_table.add_row(
        "Confidence Score",
        f"{v1_conf:.4f}",
        f"{v2_conf:.4f}",
        f"{v2_conf - v1_conf:+.4f}",
        style="bold green" if v2_conf > v1_conf else "bold red"
    )
    
    console.print(comparison_table)
    
    # Write results to file (V1 and V2)
    writer.file.write(f"\n## Version 1 Results (Yes/No only)\n\n")
    writer.write_original_answers(confidence_result_v1['original_answers'])
    writer.write_distractor_sets(confidence_result_v1['distractor_sets'], max_sets=2)
    writer.write_distracted_samples(confidence_result_v1['distracted_samples'], max_samples=2)
    writer.write_detailed_tracking(confidence_result_v1, confidence_result_v1['k1_samples'], confidence_result_v1['k2_samples'])
    writer.write_metrics_table(expert_result_str, confidence_result_v1)
    
    writer.file.write(f"\n## Version 2 Results (Yes/No + Full Explanation)\n\n")
    writer.write_distracted_samples(confidence_result_v2['distracted_samples'], max_samples=2)
    writer.write_detailed_tracking(confidence_result_v2, confidence_result_v2['k1_samples'], confidence_result_v2['k2_samples'])
    writer.write_metrics_table(expert_result_str, confidence_result_v2)
    
    console.print()
    return {'v1': confidence_result_v1, 'v2': confidence_result_v2}


def main():
    """Run all tests."""
    # Setup logging
    setup_logging(level="INFO", date_format='%H:%M:%S')
    
    # Create markdown folder if it doesn't exist
    markdown_folder = Path("markdown")
    markdown_folder.mkdir(exist_ok=True)
    
    # Create report file in markdown folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = markdown_folder / f"distractor_test_report_{timestamp}.md"
    writer = MarkdownReportWriter(str(report_filename))
    
    console.print(Panel.fit(
        "[bold cyan]Distractor Confidence Estimator Test Suite[/bold cyan]\n"
        f"Report: {report_filename}",
        border_style="cyan"
    ))
    
    # Setup LLM client
    client = OnlineLLMClient(
        api_key="sk-fnUHDzxXAimEnYgyX20Jag",
        base_url="https://llmapi.paratera.com/v1/",
        model_name="Qwen3-Next-80B-A3B-Thinking",
        max_tokens=800,
        temperature=0.7
    )
    
    # Create both estimators
    estimator_v1 = create_distractor_confidence_estimator(
        client=client,
        k1_samples=5,
        k2_samples=2,
        gamma1=1e-6,
        gamma2=1e-6,
        seed=42,
        max_workers=10
    )
    
    estimator_v2 = create_distractor_confidence_estimator_v2(
        client=client,
        k1_samples=5,
        k2_samples=2,
        gamma1=1e-6,
        gamma2=1e-6,
        seed=42,
        max_workers=10
    )
    
    # Write configuration
    writer.write_header()
    writer.write_config(client, estimator_v1)
    
    console.print(f"[bold]Configuration:[/bold] k1={estimator_v1.k1_samples}, k2={estimator_v1.k2_samples}, "
                 f"total={estimator_v1.k1_samples * estimator_v1.k2_samples * 3} samples\n")
    
    # Run tests
    try:
        # Test 1: Backdoor Path
        run_test(
            1, "Backdoor Path Detection",
            "Testing independence after blocking confounders",
            BackdoorPathExpert,
            {
                'base_prompt': "",
                'x1': "Ice Cream Sales",
                'x2': "Drowning Incidents",
                'all_variables': ["Ice Cream Sales", "Drowning Incidents", "Temperature"],
                'expert_type': "domain_knowledge"
            },
            client, estimator_v1, estimator_v2, writer
        )
        
        # Test 2: Independence
        run_test(
            2, "Statistical Independence",
            "Testing independence with known confounder",
            IndependenceExpert,
            {
                'base_prompt': "",
                'x1': "Height",
                'x2': "Reading Ability",
                'all_variables': ["Height", "Reading Ability", "Age"],
                'expert_type': "statistical"
            },
            client, estimator_v1, estimator_v2, writer
        )
        
        # Test 3: Clear Causal Direction
        run_test(
            3, "Clear Causal Direction",
            "Testing well-established causal relationship",
            CausalDirectionExpert,
            {
                'base_prompt': "",
                'x1': "Smoking",
                'x2': "Lung Cancer",
                'all_variables': ["Smoking", "Lung Cancer", "Age"],
                'expert_type': "domain_knowledge"
            },
            client, estimator_v1, estimator_v2, writer
        )
        
        # Test 4: Spurious Correlation
        run_test(
            4, "Spurious Correlation",
            "Testing spurious correlation scenario",
            CausalDirectionExpert,
            {
                'base_prompt': "",
                'x1': "Rooster Crowing",
                'x2': "Sunrise",
                'all_variables': ["Rooster Crowing", "Sunrise", "Time of Day"],
                'expert_type': "temporal_dynamics"
            },
            client, estimator_v1, estimator_v2, writer
        )
        
        # Test 5: Latent Confounder
        run_test(
            5, "Latent Confounder Detection",
            "Testing for hidden confounders",
            LatentConfounderExpert,
            {
                'base_prompt': "",
                'x1': "Exercise",
                'x2': "Weight Loss",
                'all_variables': ["Exercise", "Weight Loss", "Diet"],
                'expert_type': "domain_knowledge"
            },
            client, estimator_v1, estimator_v2, writer
        )
        
        console.print(Panel.fit(
            f"[bold green]✓ All tests completed successfully![/bold green]\n"
            f"Report saved to: {report_filename}",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
