"""
Output manager for saving analysis results.
Organizes outputs by category in output/analysis directory.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "analysis"


class OutputManager:
    """
    Manages output files organized by category.

    Directory structure:
    output/analysis/
    ├── trends/           # Trend analysis results
    │   └── {conference}_{timestamp}.json
    ├── topics/          # Topic modeling results
    ├── ecosystem/       # Ecosystem analysis results
    │   ├── vocabulary_timeline.csv
    │   ├── similarity_matrix.csv
    │   ├── diffusion_paths.json
    │   └── knowledge_flow.json
    ├── network/         # Network analysis results
    │   └── {analysis_type}_{timestamp}.json
    ├── deep/            # Deep domain analysis
    │   └── {domain}_{timestamp}.json
    └── reports/         # Generated reports
        └── {report_type}_{timestamp}.json
    """

    CATEGORIES = [
        'trends', 'topics', 'ecosystem', 'network', 'deep', 'reports'
    ]

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or DEFAULT_OUTPUT_DIR
        self._ensure_directories()

    def _ensure_directories(self):
        """Create category directories if they don't exist."""
        for category in self.CATEGORIES:
            (self.base_dir / category).mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_json(
        self,
        data: Any,
        category: str,
        filename: str = None,
        indent: int = 2
    ) -> Path:
        """
        Save data as JSON file.

        Args:
            data: Data to save
            category: Category folder (trends/topics/ecosystem/network/deep/reports)
            filename: Custom filename (auto-generated if None)
            indent: JSON indentation

        Returns:
            Path to saved file
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        category_dir = self.base_dir / category

        if filename is None:
            timestamp = self._get_timestamp()
            filename = f"{category}_{timestamp}.json"

        filepath = category_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return filepath

    def save_csv(
        self,
        data: List[Dict],
        category: str,
        filename: str = None
    ) -> Path:
        """
        Save data as CSV file.

        Args:
            data: List of dictionaries to save as CSV
            category: Category folder
            filename: Custom filename

        Returns:
            Path to saved file
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        category_dir = self.base_dir / category

        if filename is None:
            timestamp = self._get_timestamp()
            filename = f"{category}_{timestamp}.csv"

        filepath = category_dir / filename

        if not data:
            filepath.touch()
            return filepath

        fieldnames = list(data[0].keys())
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        return filepath

    def save_report(
        self,
        report: Dict,
        report_type: str,
        domain: str = None
    ) -> Path:
        """
        Save analysis report with standardized naming.

        Args:
            report: Report data
            report_type: Type of report (trend, ecosystem, network, deep)
            domain: Optional domain name

        Returns:
            Path to saved report
        """
        timestamp = self._get_timestamp()

        if domain:
            filename = f"{report_type}_{domain.replace(' ', '_')}_{timestamp}.json"
        else:
            filename = f"{report_type}_{timestamp}.json"

        return self.save_json(report, 'reports', filename)

    def get_category_path(self, category: str) -> Path:
        """Get path to category directory."""
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}")
        return self.base_dir / category

    def list_outputs(self, category: str = None) -> Dict[str, List[Path]]:
        """
        List all output files.

        Args:
            category: Filter by category (None for all)

        Returns:
            Dictionary mapping category to list of file paths
        """
        if category:
            if category not in self.CATEGORIES:
                raise ValueError(f"Invalid category: {category}")
            return {category: list((self.base_dir / category).glob('*'))}

        return {
            cat: list((self.base_dir / cat).glob('*'))
            for cat in self.CATEGORIES
        }


# Default instance for quick access
default_output = OutputManager()
