"""
Report Logger for Recording LLM Prompts and Responses

This module provides functionality to record all LLM prompts and responses
to markdown files in the /running_reports folder, similar to the existing
causal discovery report format.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from threading import Lock

logger = logging.getLogger(__name__)


class ReportLogger:
    """
    Logger for recording LLM prompts and responses to markdown files.
    
    This class maintains a single report file per test run and records
    all LLM interactions in a structured markdown format.
    """
    
    def __init__(self, report_dir: str = "running_reports", enabled: bool = True):
        """
        Initialize the report logger.
        
        Args:
            report_dir: Directory to save report files
            enabled: Whether logging is enabled
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.report_file: Optional[Path] = None
        self.interaction_count = 0
        self._lock = Lock()
        
        logger.info(f"ReportLogger initialized (enabled={enabled}, dir={report_dir})")
    
    def start_new_report(
        self,
        report_name: str,
        config_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Start a new report file.
        
        Args:
            report_name: Name/title for the report
            config_info: Optional configuration information to include
        
        Returns:
            Path to the created report file
        """
        if not self.enabled:
            return None
        
        with self._lock:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name}_{timestamp}.md"
            self.report_file = self.report_dir / filename
            self.interaction_count = 0
            
            # Write report header
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(f"# LLM Interaction Report - {report_name}\n\n")
                f.write(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if config_info:
                    f.write("---\n\n## Configuration\n\n")
                    for key, value in config_info.items():
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")
                
                f.write("---\n\n## LLM Interactions\n\n")
            
            logger.info(f"Started new report: {self.report_file}")
            return self.report_file
    
    def log_interaction(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single LLM interaction (prompt and response).
        
        Args:
            prompt: The prompt sent to the LLM
            response: The response received from the LLM
            context: Optional context description for this interaction
            metadata: Optional metadata (model, temperature, etc.)
        """
        if not self.enabled or self.report_file is None:
            return
        
        with self._lock:
            self.interaction_count += 1
            
            with open(self.report_file, 'a', encoding='utf-8') as f:
                # Write interaction header
                f.write(f"### Interaction {self.interaction_count}\n\n")
                
                if context:
                    f.write(f"**Context:** {context}\n\n")
                
                if metadata:
                    f.write("**Metadata:**\n\n")
                    for key, value in metadata.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                # Write prompt
                f.write("**Prompt:**\n\n")
                f.write("```\n")
                f.write(prompt)
                f.write("\n```\n\n")
                
                # Write response
                f.write("**Response:**\n\n")
                f.write("```\n")
                f.write(response)
                f.write("\n```\n\n")
                
                f.write("---\n\n")
    
    def log_expert_interaction(
        self,
        expert_name: str,
        expert_type: str,
        prompt: str,
        response: str,
        result: Optional[str] = None,
        pair_info: Optional[str] = None
    ):
        """
        Log an expert-specific interaction with specialized formatting.
        
        Args:
            expert_name: Name of the expert
            expert_type: Type of expert (e.g., 'graph_theory', 'independence')
            prompt: The prompt sent to the expert
            response: The response from the expert
            result: Optional parsed result (Yes/No, etc.)
            pair_info: Optional variable pair information
        """
        if not self.enabled or self.report_file is None:
            return
        
        with self._lock:
            self.interaction_count += 1
            
            with open(self.report_file, 'a', encoding='utf-8') as f:
                # Write expert interaction header
                f.write(f"### Interaction {self.interaction_count}: Expert - {expert_name}\n\n")
                
                if pair_info:
                    f.write(f"**Variable Pair:** {pair_info}\n\n")
                
                f.write(f"**Expert Type:** {expert_type}\n\n")
                
                if result:
                    f.write(f"**Result:** {result}\n\n")
                
                # Write prompt
                f.write("**Prompt:**\n\n")
                f.write("```\n")
                f.write(prompt)
                f.write("\n```\n\n")
                
                # Write response
                f.write("**Response:**\n\n")
                f.write("```\n")
                f.write(response)
                f.write("\n```\n\n")
                
                f.write("---\n\n")
    
    def add_section(self, title: str, content: str):
        """
        Add a custom section to the report.
        
        Args:
            title: Section title
            content: Section content
        """
        if not self.enabled or self.report_file is None:
            return
        
        with self._lock:
            with open(self.report_file, 'a', encoding='utf-8') as f:
                f.write(f"## {title}\n\n")
                f.write(content)
                f.write("\n\n---\n\n")
    
    def close_report(self, summary: Optional[str] = None):
        """
        Close the current report and optionally add a summary.
        
        Args:
            summary: Optional summary text to append at the end
        """
        if not self.enabled or self.report_file is None:
            return
        
        with self._lock:
            with open(self.report_file, 'a', encoding='utf-8') as f:
                if summary:
                    f.write("## Summary\n\n")
                    f.write(summary)
                    f.write("\n\n")
                
                f.write("---\n\n")
                f.write(f"**Report completed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total interactions logged:** {self.interaction_count}\n")
            
            logger.info(f"Report closed: {self.report_file} ({self.interaction_count} interactions)")
            self.report_file = None
            self.interaction_count = 0


# Global report logger instance
_global_report_logger: Optional[ReportLogger] = None


def get_report_logger() -> ReportLogger:
    """
    Get the global report logger instance.
    
    Returns:
        Global ReportLogger instance
    """
    global _global_report_logger
    if _global_report_logger is None:
        _global_report_logger = ReportLogger()
    return _global_report_logger


def set_report_logger(logger: ReportLogger):
    """
    Set the global report logger instance.
    
    Args:
        logger: ReportLogger instance to use globally
    """
    global _global_report_logger
    _global_report_logger = logger


def enable_report_logging(enabled: bool = True):
    """
    Enable or disable report logging globally.
    
    Args:
        enabled: Whether to enable logging
    """
    logger = get_report_logger()
    logger.enabled = enabled
