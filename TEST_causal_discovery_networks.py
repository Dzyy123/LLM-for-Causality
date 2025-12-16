"""
Test Causal Discovery Framework

Runs causal discovery on variable sets and outputs results to Markdown files.
Uses logging for progress tracking and rich formatting for console output.
Records ALL prompts and responses with numbered subtitles.
"""

import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from tree_query import CausalDiscoveryFramework
from llm_utils import OnlineLLMClient, setup_logging


# Initialize rich console
console = Console()


class CausalDiscoveryReportWriter:
    """Writes causal discovery results in Markdown format with ALL prompts and responses."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.prompt_counter = 0  # Global counter for all prompts/responses
    
    def write_header(self, variables: List[str]):
        """Write report header."""
        self.file.write("# Causal Discovery Framework Test Report\n\n")
        self.file.write(f"**Test Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file.write(f"**Variables:** {', '.join(variables)}\n\n")
        self.file.write("---\n\n")
        self.flush()
    
    def write_config(self, client, framework: CausalDiscoveryFramework):
        """Write configuration section."""
        self.file.write("## Configuration\n\n")
        self.file.write("### LLM Client\n\n")
        self.file.write(f"- **API Base:** {client.base_url}\n")
        self.file.write(f"- **Model:** {client.model_name}\n")
        self.file.write(f"- **Max Tokens:** {client.max_tokens}\n")
        self.file.write(f"- **Temperature:** {client.temperature}\n\n")
        
        self.file.write("### Framework Parameters\n\n")
        self.file.write(f"- **Variables:** {', '.join(framework.all_variables)}\n")
        self.file.write(f"- **Trust Confidence:** {framework.trust_confidence}\n")
        self.file.write(f"- **K1 Samples:** {framework.confidence_estimator.k1_samples}\n")
        self.file.write(f"- **K2 Samples:** {framework.confidence_estimator.k2_samples}\n\n")
        self.file.write("---\n\n")
        self.flush()
    
    def write_pair_header(self, pair_num: int, x1: str, x2: str):
        """Write section header for a variable pair."""
        self.file.write(f"## Pair {pair_num}: {x1} vs {x2}\n\n")
        self.flush()
    
    def write_backdoor_result(self, have_backdoor: bool, confidence: float, log: List):
        """Write backdoor path check result with ALL prompts and responses."""
        self.file.write("### Backdoor Path Check\n\n")
        self.file.write(f"- **Has Backdoor Path:** {'Yes' if have_backdoor else 'No'}\n")
        self.file.write(f"- **Confidence:** {confidence:.4f}\n\n")
        
        if log:
            for step_name, step_result in log:
                self.file.write(f"#### {step_name}\n\n")
                self.file.write(f"- **Label:** {'Yes' if step_result.get('label') == 1 else 'No'}\n")
                self.file.write(f"- **Confidence:** {step_result.get('confidence', 'N/A')}\n\n")
                
                # Write ALL expert prompts and responses
                expert_results = step_result.get('expert_results', [])
                if expert_results:
                    self.file.write("##### Expert Prompts and Responses\n\n")
                    for er in expert_results:
                        self.prompt_counter += 1
                        expert_name = er.get('expert', 'unknown')
                        expert_label = 'Yes' if er.get('label') == 1 else 'No'
                        prompt = er.get('prompt', 'N/A')
                        response = er.get('response', 'N/A')
                        
                        self.file.write(f"###### [{self.prompt_counter}] Expert: {expert_name} (Result: {expert_label})\n\n")
                        self.file.write(f"**Prompt:**\n\n")
                        self.file.write(f"```\n{prompt}\n```\n\n")
                        self.file.write(f"**Response:**\n\n")
                        self.file.write(f"```\n{response}\n```\n\n")
                
                # Write distractor confidence estimation details
                confidence_details = step_result.get('confidence_details')
                if confidence_details:
                    self._write_distractor_details(confidence_details)
        
        self.flush()
    
    def _write_distractor_details(self, confidence_details: Dict[str, Any]):
        """Write detailed distractor sets and distracted responses."""
        if not confidence_details:
            return
        
        self.file.write("**Distractor-Based Confidence Estimation Details:**\n\n")
        
        # Write summary metrics
        self.file.write(f"- **Confidence Score:** {confidence_details.get('confidence_score', 0.0):.4f}\n")
        self.file.write(f"- **Robustness Score:** {confidence_details.get('robustness_score', 0.0):.2%}\n")
        self.file.write(f"- **K1 Samples:** {confidence_details.get('k1_samples', 'N/A')}\n")
        self.file.write(f"- **K2 Samples:** {confidence_details.get('k2_samples', 'N/A')}\n")
        self.file.write(f"- **Total Distracted Samples:** {confidence_details.get('total_samples', 'N/A')}\n\n")
        
        # Write flip rates
        flip_rates = confidence_details.get('flip_rates', {})
        if flip_rates:
            self.file.write("**Flip Rates by Distractor Type:**\n\n")
            self.file.write("| Distractor Type | Flip Rate |\n")
            self.file.write("|-----------------|----------|\n")
            for dtype, rate in flip_rates.items():
                self.file.write(f"| {dtype} | {rate:.2%} |\n")
            self.file.write("\n")
        
        # Write distractor sets
        distractor_sets = confidence_details.get('distractor_sets', [])
        if distractor_sets:
            self.file.write(f"**Distractor Sets ({len(distractor_sets)} sets):**\n\n")
            
            for i, dset in enumerate(distractor_sets[:5], 1):  # Limit to first 5 for brevity
                self.prompt_counter += 1
                self.file.write(f"<details>\n<summary>[{self.prompt_counter}] Distractor Set {dset.set_index + 1} (Original Label: {'Yes' if dset.original_label == 1 else 'No'})</summary>\n\n")
                
                # Original answer
                self.file.write(f"**Original Answer:**\n\n```\n{dset.original_answer[:500]}{'...' if len(dset.original_answer) > 500 else ''}\n```\n\n")
                
                # Contrarian distractor
                self.file.write(f"**Contrarian Distractor:**\n\n")
                self.file.write(f"*Generation Prompt:*\n```\n{dset.contrarian.prompt[:300]}{'...' if len(dset.contrarian.prompt) > 300 else ''}\n```\n\n")
                self.file.write(f"*Generated Argument:*\n```\n{dset.contrarian.argument[:500]}{'...' if len(dset.contrarian.argument) > 500 else ''}\n```\n\n")
                
                # Deceiver distractor
                self.file.write(f"**Deceiver Distractor:**\n\n")
                self.file.write(f"*Generation Prompt:*\n```\n{dset.deceiver.prompt[:300]}{'...' if len(dset.deceiver.prompt) > 300 else ''}\n```\n\n")
                self.file.write(f"*Generated Argument:*\n```\n{dset.deceiver.argument[:500]}{'...' if len(dset.deceiver.argument) > 500 else ''}\n```\n\n")
                
                # Hater distractor
                self.file.write(f"**Hater Distractor:**\n\n")
                self.file.write(f"*Generation Prompt:*\n```\n{dset.hater.prompt[:300]}{'...' if len(dset.hater.prompt) > 300 else ''}\n```\n\n")
                self.file.write(f"*Generated Argument:*\n```\n{dset.hater.argument[:500]}{'...' if len(dset.hater.argument) > 500 else ''}\n```\n\n")
                
                self.file.write("</details>\n\n")
            
        # Write distracted samples
        distracted_samples = confidence_details.get('distracted_samples', [])
        if distracted_samples:
            self.file.write(f"**Distracted Responses ({len(distracted_samples)} samples):**\n\n")
            
            # Summary table
            self.file.write("| Set | Type | Original | Distracted | Flipped |\n")
            self.file.write("|-----|------|----------|------------|--------|\n")
            
            for sample in distracted_samples[:15]:  # Limit to first 15
                orig_label = 'Yes' if sample.original_label == 1 else 'No'
                dist_label = 'Yes' if sample.label == 1 else 'No'
                flipped = '✓' if sample.label != sample.original_label else ''
                self.file.write(f"| {sample.set_index + 1} | {sample.distractor_type} | {orig_label} | {dist_label} | {flipped} |\n")
            
            self.file.write("\n")
            
            # Detailed distracted samples (first 3)
            self.file.write("**Sample Distracted Response Details:**\n\n")
            for sample in distracted_samples[:3]:
                self.prompt_counter += 1
                self.file.write(f"<details>\n<summary>[{self.prompt_counter}] Set {sample.set_index + 1} - {sample.distractor_type} (Result: {'Yes' if sample.label == 1 else 'No'})</summary>\n\n")
                self.file.write(f"**Sampling Prompt:**\n\n```\n{sample.sampling_prompt}\n```\n\n")
                self.file.write(f"**Response:**\n\n```\n{sample.response}\n```\n\n")
                self.file.write("</details>\n\n")
            
        self.flush()
    
    def _write_branch_statistics_table(self, results: List[Dict[str, Any]]):
        """Write a very detailed statistics table for all explored branches."""
        
        # Collect all statistics
        all_branch_stats = []
        
        for branch_idx, result in enumerate(results, 1):
            branch_relation = result.get('relation', 'N/A')
            branch_confidence = result.get('confidence', 0.0)
            log = result.get('log', [])
            
            for step_idx, (step_name, step_result) in enumerate(log, 1):
                step_label = step_result.get('label', 0)
                step_confidence = step_result.get('confidence', 0.0)
                expert_results = step_result.get('expert_results', [])
                
                # Calculate expert statistics
                total_experts = len(expert_results)
                yes_votes = sum(1 for er in expert_results if er.get('label') == 1)
                no_votes = total_experts - yes_votes
                yes_ratio = yes_votes / total_experts if total_experts > 0 else 0.0
                
                # Get expert names
                expert_names = [er.get('expert', 'unknown') for er in expert_results]
                yes_experts = [er.get('expert', 'unknown') for er in expert_results if er.get('label') == 1]
                no_experts = [er.get('expert', 'unknown') for er in expert_results if er.get('label') == 0]
                
                all_branch_stats.append({
                    'branch_idx': branch_idx,
                    'branch_relation': branch_relation,
                    'branch_confidence': branch_confidence,
                    'step_idx': step_idx,
                    'step_name': step_name,
                    'step_label': step_label,
                    'step_confidence': step_confidence,
                    'total_experts': total_experts,
                    'yes_votes': yes_votes,
                    'no_votes': no_votes,
                    'yes_ratio': yes_ratio,
                    'expert_names': expert_names,
                    'yes_experts': yes_experts,
                    'no_experts': no_experts
                })
        
        if not all_branch_stats:
            self.file.write("*No branch statistics available.*\n\n")
            return
        
        # Write comprehensive statistics table
        self.file.write("##### Per-Step Voting Statistics\n\n")
        self.file.write("| Branch | Step | Step Name | Decision | Confidence | Yes | No | Yes% | Consensus |\n")
        self.file.write("|--------|------|-----------|----------|------------|-----|-----|------|----------|\n")
        
        for stat in all_branch_stats:
            branch_id = f"B{stat['branch_idx']}"
            step_id = f"S{stat['step_idx']}"
            step_name = stat['step_name'][:30] + "..." if len(stat['step_name']) > 30 else stat['step_name']
            decision = "Yes" if stat['step_label'] == 1 else "No"
            confidence = f"{stat['step_confidence']:.4f}"
            yes_votes = stat['yes_votes']
            no_votes = stat['no_votes']
            yes_pct = f"{stat['yes_ratio']*100:.1f}%"
            
            # Consensus indicator
            if stat['yes_ratio'] >= 0.9:
                consensus = "Strong Yes"
            elif stat['yes_ratio'] <= 0.1:
                consensus = "Strong No"
            elif stat['yes_ratio'] >= 0.7:
                consensus = "Lean Yes"
            elif stat['yes_ratio'] <= 0.3:
                consensus = "Lean No"
            else:
                consensus = "Split"
            
            self.file.write(f"| {branch_id} | {step_id} | {step_name} | {decision} | {confidence} | {yes_votes} | {no_votes} | {yes_pct} | {consensus} |\n")
        
        self.file.write("\n")
        
        # Write expert breakdown table
        self.file.write("##### Expert Vote Breakdown by Step\n\n")
        self.file.write("| Branch | Step | Step Name | Yes Experts | No Experts |\n")
        self.file.write("|--------|------|-----------|-------------|------------|\n")
        
        for stat in all_branch_stats:
            branch_id = f"B{stat['branch_idx']}"
            step_id = f"S{stat['step_idx']}"
            step_name = stat['step_name'][:20] + "..." if len(stat['step_name']) > 20 else stat['step_name']
            yes_experts = ", ".join(stat['yes_experts'][:3]) if stat['yes_experts'] else "-"
            if len(stat['yes_experts']) > 3:
                yes_experts += f" (+{len(stat['yes_experts'])-3})"
            no_experts = ", ".join(stat['no_experts'][:3]) if stat['no_experts'] else "-"
            if len(stat['no_experts']) > 3:
                no_experts += f" (+{len(stat['no_experts'])-3})"
            
            self.file.write(f"| {branch_id} | {step_id} | {step_name} | {yes_experts} | {no_experts} |\n")
        
        self.file.write("\n")
        
        # Write branch summary statistics
        self.file.write("##### Branch Summary Statistics\n\n")
        
        # Group by branch
        branches = {}
        for stat in all_branch_stats:
            bid = stat['branch_idx']
            if bid not in branches:
                branches[bid] = {
                    'relation': stat['branch_relation'],
                    'confidence': stat['branch_confidence'],
                    'steps': [],
                    'total_yes': 0,
                    'total_no': 0,
                    'total_experts': 0
                }
            branches[bid]['steps'].append(stat)
            branches[bid]['total_yes'] += stat['yes_votes']
            branches[bid]['total_no'] += stat['no_votes']
            branches[bid]['total_experts'] += stat['total_experts']
        
        self.file.write("| Branch | Relation | Final Confidence | Steps | Total Votes | Avg Yes% | Path |\n")
        self.file.write("|--------|----------|------------------|-------|-------------|----------|------|\n")
        
        for bid, bdata in branches.items():
            total_votes = bdata['total_yes'] + bdata['total_no']
            avg_yes_pct = (bdata['total_yes'] / total_votes * 100) if total_votes > 0 else 0.0
            
            # Build path string
            path_parts = []
            for s in bdata['steps']:
                decision = "Y" if s['step_label'] == 1 else "N"
                path_parts.append(f"{s['step_name'][:10]}={decision}")
            path_str = " → ".join(path_parts)
            if len(path_str) > 50:
                path_str = path_str[:47] + "..."
            
            self.file.write(f"| B{bid} | {bdata['relation']} | {bdata['confidence']:.4f} | {len(bdata['steps'])} | {total_votes} | {avg_yes_pct:.1f}% | {path_str} |\n")
        
        self.file.write("\n")
        
        # Write overall statistics
        self.file.write("##### Overall Statistics\n\n")
        total_branches = len(branches)
        total_steps = len(all_branch_stats)
        total_expert_calls = sum(s['total_experts'] for s in all_branch_stats)
        total_yes_votes = sum(s['yes_votes'] for s in all_branch_stats)
        total_no_votes = sum(s['no_votes'] for s in all_branch_stats)
        overall_yes_pct = (total_yes_votes / (total_yes_votes + total_no_votes) * 100) if (total_yes_votes + total_no_votes) > 0 else 0.0
        
        # Count unique relations found
        relations_found = set(b['relation'] for b in branches.values())
        
        self.file.write(f"- **Total Branches Explored:** {total_branches}\n")
        self.file.write(f"- **Total Steps Executed:** {total_steps}\n")
        self.file.write(f"- **Total Expert Calls:** {total_expert_calls}\n")
        self.file.write(f"- **Total Yes Votes:** {total_yes_votes}\n")
        self.file.write(f"- **Total No Votes:** {total_no_votes}\n")
        self.file.write(f"- **Overall Yes Rate:** {overall_yes_pct:.2f}%\n")
        self.file.write(f"- **Unique Relations Found:** {', '.join(relations_found)}\n\n")
        
        self.flush()
    
    def write_explored_results(self, results: List[Dict[str, Any]]):
        """Write all explored results from tree query with ALL prompts and responses."""
        self.file.write("### Explored Results\n\n")
        self.file.write(f"**Total Results Explored:** {len(results)}\n\n")
        
        if not results:
            self.file.write("*No results explored.*\n\n")
            return
        
        # Summary table
        self.file.write("#### Summary Table\n\n")
        self.file.write("| # | Relation | Confidence | Steps |\n")
        self.file.write("|---|----------|------------|-------|\n")
        
        for i, result in enumerate(results, 1):
            relation = result.get('relation', 'N/A')
            confidence = result.get('confidence', 0.0)
            steps = len(result.get('log', []))
            self.file.write(f"| {i} | {relation} | {confidence:.4f} | {steps} |\n")
        
        self.file.write("\n")
        
        # Detailed statistics table for each branch
        self.file.write("#### Detailed Branch Statistics\n\n")
        self._write_branch_statistics_table(results)
        
        # Detailed logs for each result with ALL prompts and responses
        for i, result in enumerate(results, 1):
            self.file.write(f"#### Result {i}: {result.get('relation', 'N/A')} (Confidence: {result.get('confidence', 0.0):.4f})\n\n")
            
            for step_name, step_result in result.get('log', []):
                self.file.write(f"##### Step: {step_name}\n\n")
                self.file.write(f"- **Label:** {'Yes' if step_result.get('label') == 1 else 'No'}\n")
                self.file.write(f"- **Confidence:** {step_result.get('confidence', 'N/A'):.4f}\n\n")
                
                # Write ALL expert prompts and responses
                expert_results = step_result.get('expert_results', [])
                if expert_results:
                    self.file.write(f"**Expert Prompts and Responses ({len(expert_results)} experts):**\n\n")
                    for er in expert_results:
                        self.prompt_counter += 1
                        expert_name = er.get('expert', 'unknown')
                        expert_label = 'Yes' if er.get('label') == 1 else 'No'
                        prompt = er.get('prompt', 'N/A')
                        response = er.get('response', 'N/A')
                        
                        self.file.write(f"###### [{self.prompt_counter}] Expert: {expert_name} (Result: {expert_label})\n\n")
                        self.file.write(f"**Prompt:**\n\n")
                        self.file.write(f"```\n{prompt}\n```\n\n")
                        self.file.write(f"**Response:**\n\n")
                        self.file.write(f"```\n{response}\n```\n\n")
                
                # Write distractor confidence estimation details
                confidence_details = step_result.get('confidence_details')
                if confidence_details:
                    self._write_distractor_details(confidence_details)
        
        self.flush()
    
    def write_resolved_result(self, resolved: Dict[str, Any]):
        """Write the final resolved result."""
        self.file.write("### Final Resolved Result\n\n")
        self.file.write(f"- **Relation:** `{resolved.get('relation', 'N/A')}`\n")
        self.file.write(f"- **Confidence:** {resolved.get('confidence', 0.0):.4f}\n")
        self.file.write(f"- **Has Backdoor:** {'Yes' if resolved.get('have_backdoor') else 'No'}\n")
        self.file.write(f"- **Backdoor Confidence:** {resolved.get('backdoor_confidence', 0.0):.4f}\n")
        self.file.write(f"- **Total Alternatives Explored:** {len(resolved.get('all_results', []))}\n\n")
        self.file.write("---\n\n")
        self.flush()
    
    def write_summary_table(self, all_relations: Dict[Tuple[str, str], Dict[str, Any]]):
        """Write summary table of all discovered relations."""
        self.file.write("## Summary of All Relations\n\n")
        
        self.file.write("| Variable 1 | Variable 2 | Relation | Confidence | Backdoor |\n")
        self.file.write("|------------|------------|----------|------------|----------|\n")
        
        for (x1, x2), result in all_relations.items():
            relation = result.get('relation', 'N/A')
            confidence = result.get('confidence', 0.0)
            backdoor = 'Yes' if result.get('have_backdoor') else 'No'
            self.file.write(f"| {x1} | {x2} | {relation} | {confidence:.4f} | {backdoor} |\n")
        
        self.file.write("\n")
        self.file.write(f"**Total Prompts/Responses Recorded:** {self.prompt_counter}\n\n")
        self.flush()
    

    
    def write_prior_graph(self, prior_graph: Dict[Tuple[str, str], str]):
        """Write prior causal graph."""
        self.file.write("## Prior Causal Graph\n\n")
        
        self.file.write("```\n")
        for pair, relation in prior_graph.items():
            self.file.write(f"{pair[0]} {relation} {pair[1]}\n")
        self.file.write("```\n\n")
        self.flush()
    
    def flush(self):
        """Flush file buffer."""
        self.file.flush()
    
    def close(self):
        """Close file."""
        self.file.write("\n---\n\n")
        self.file.write(f"**Total Prompts/Responses in Report:** {self.prompt_counter}\n\n")
        self.file.write("*End of Report*\n")
        self.file.close()


def display_pair_result(x1: str, x2: str, query_results: Dict[str, Any], resolved: Dict[str, Any]):
    """Display results for a variable pair in console."""
    console.print(f"\n[bold cyan]Analyzing: {x1} vs {x2}[/bold cyan]")
    
    # Backdoor info
    backdoor_str = "[green]Yes[/green]" if query_results['have_backdoor'] else "[red]No[/red]"
    console.print(f"  Backdoor Path: {backdoor_str} (confidence: {query_results['backdoor_confidence']:.4f})")
    
    # All explored results
    results_table = Table(title="Explored Results", box=box.SIMPLE)
    results_table.add_column("#", style="dim", width=3)
    results_table.add_column("Relation", style="cyan")
    results_table.add_column("Confidence", style="yellow", justify="right")
    
    for i, result in enumerate(query_results['results'], 1):
        conf_style = "green" if result['confidence'] >= 0.75 else "yellow" if result['confidence'] >= 0.5 else "red"
        results_table.add_row(
            str(i),
            result['relation'],
            f"[{conf_style}]{result['confidence']:.4f}[/{conf_style}]"
        )
    
    console.print(results_table)
    
    # Final result
    console.print(f"  [bold]Final: {resolved['relation']}[/bold] (confidence: {resolved['confidence']:.4f})")


def display_summary_table(all_relations: Dict[Tuple[str, str], Dict[str, Any]]):
    """Display summary table in console."""
    table = Table(title="Causal Discovery Results Summary", box=box.ROUNDED)
    table.add_column("Variable 1", style="cyan")
    table.add_column("Variable 2", style="cyan")
    table.add_column("Relation", style="bold")
    table.add_column("Confidence", style="yellow", justify="right")
    table.add_column("Backdoor", justify="center")
    
    for (x1, x2), result in all_relations.items():
        relation = result.get('relation', 'N/A')
        confidence = result.get('confidence', 0.0)
        backdoor = "[green]Yes[/green]" if result.get('have_backdoor') else "[red]No[/red]"
        
        conf_style = "green" if confidence >= 0.75 else "yellow" if confidence >= 0.5 else "red"
        
        table.add_row(
            x1,
            x2,
            relation,
            f"[{conf_style}]{confidence:.4f}[/{conf_style}]",
            backdoor
        )
    
    console.print(table)


def main():
    """Run causal discovery and output results."""
    # Setup logging
    setup_logging(level="INFO", date_format='%H:%M:%S')
    
    # Create markdown folder if it doesn't exist
    markdown_folder = Path("running_reports")
    markdown_folder.mkdir(exist_ok=True)
    
    # Create report file in running_reports folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = markdown_folder / f"causal_discovery_report_{timestamp}.md"
    writer = CausalDiscoveryReportWriter(str(report_filename))
    
    console.print(Panel.fit(
        "[bold cyan]Causal Discovery Framework Test[/bold cyan]\n"
        f"Report: {report_filename}",
        border_style="cyan"
    ))
    
    # Initialize LLM client
    client = OnlineLLMClient(
        api_key="sk-3cjup3m-TnnjBxc1Vn7uIw",
        base_url="https://llmapi.paratera.com/v1/",
        model_name="DeepSeek-V3.2-Instruct",
        max_tokens=800,
        temperature=0.7
    )
    
    # Define variables
    variables = ["Ice Cream Sales", "Drowning Deaths", "Temperature"]
    
    # Create framework
    framework = CausalDiscoveryFramework(
        client=client,
        all_variables=variables,
        k1_samples=10,
        k2_samples=1,
        seed=42,
        max_workers=30,
        trust_confidence=0.75
    )
    
    # Write report header and config
    writer.write_header(variables)
    writer.write_config(client, framework)
    
    # Store all results for summary
    all_resolved_relations = {}
    
    # Analyze each pair
    console.rule("[bold blue]Analyzing Variable Pairs")
    
    from itertools import combinations
    pair_num = 0
    for x1, x2 in combinations(variables, 2):
        pair_num += 1
        
        # Run tree query
        query_results = framework.tree_query(x1, x2)
        resolved = framework.resolve_query_results(query_results)
        all_resolved_relations[(x1, x2)] = resolved
        
        # Display in console
        display_pair_result(x1, x2, query_results, resolved)
        
        # Write to report
        writer.write_pair_header(pair_num, x1, x2)
        writer.write_backdoor_result(
            query_results['have_backdoor'],
            query_results['backdoor_confidence'],
            query_results['backdoor_log']
        )
        writer.write_explored_results(query_results['results'])
        writer.write_resolved_result(resolved)
    
    # Summary
    console.rule("[bold blue]Summary")
    display_summary_table(all_resolved_relations)
    writer.write_summary_table(all_resolved_relations)
    
    # Create prior graph directly from resolved relations
    prior_graph = framework.create_prior_causal_graph(all_resolved_relations)
    writer.write_prior_graph(prior_graph)
    
    # Display prior graph
    console.print("\n[bold]Prior Causal Graph:[/bold]")
    for pair, relation in prior_graph.items():
        console.print(f"  {pair[0]} [cyan]{relation}[/cyan] {pair[1]}")
    
    # Close report
    writer.close()
    console.print(f"\n[green]Report saved to: {report_filename}[/green]")
    console.print(f"[dim]Total prompts/responses recorded: {writer.prompt_counter}[/dim]")


# Example usage
if __name__ == "__main__":
    main()
