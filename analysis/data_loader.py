#!/usr/bin/env python3
"""
Paper data loader module.
Load and manage papers from JSON files.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Paper data structure."""
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    dblp_key: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'abstract': self.abstract,
            'doi': self.doi,
            'url': self.url,
            'dblp_key': self.dblp_key
        }

    @property
    def has_abstract(self) -> bool:
        """Check if paper has abstract."""
        return bool(self.abstract and self.abstract.strip())

    @property
    def text_content(self) -> str:
        """Get combined text content for analysis."""
        parts = [self.title]
        if self.has_abstract:
            parts.append(self.abstract)
        return " ".join(parts)


@dataclass
class ConferenceData:
    """Conference data container."""
    name: str  # e.g., 'aaai', 'nips', 'acl'
    display_name: str  # e.g., 'AAAI', 'NeurIPS'
    papers: List[Paper] = field(default_factory=list)
    year_range: Tuple[int, int] = (0, 0)

    @property
    def years(self) -> List[int]:
        """Get list of years."""
        return sorted(set(p.year for p in self.papers))

    @property
    def paper_count(self) -> int:
        """Get total paper count."""
        return len(self.papers)

    @property
    def papers_with_abstract(self) -> int:
        """Get count of papers with abstract."""
        return sum(1 for p in self.papers if p.has_abstract)

    def get_papers_by_year(self, year: int) -> List[Paper]:
        """Get papers by year."""
        return [p for p in self.papers if p.year == year]

    def get_papers_by_year_range(self, start_year: int, end_year: int) -> List[Paper]:
        """Get papers by year range."""
        return [p for p in self.papers if start_year <= p.year <= end_year]


class PaperDataLoader:
    """Load paper data from JSON files."""

    def __init__(self, data_dir: str = None):
        """
        Initialize data loader.

        Args:
            data_dir: Path to data directory. Defaults to project data/paper directory.
        """
        if data_dir is None:
            # Default to project data directory
            root_dir = Path(__file__).parent.parent
            data_dir = root_dir / "data" / "paper"

        self.data_dir = Path(data_dir)
        logger.info(f"Data directory: {self.data_dir}")

        # Conference name mapping
        self.conf_name_mapping = {
            'aaai': 'AAAI',
            'nips': 'NeurIPS',
            'neurips': 'NeurIPS',
            'acl': 'ACL',
            'cvpr': 'CVPR',
            'iccv': 'ICCV',
            'icml': 'ICML',
            'ijcai': 'IJCAI',
            'iclr': 'ICLR',
            'kdd': 'KDD',
            'sigmod': 'SIGMOD',
            'icde': 'ICDE',
            'sigir': 'SIGIR',
            'mm': 'ACM Multimedia',
            'em': 'EMNLP',
            'emnlp': 'EMNLP',
            'naacl': 'NAACL',
            'coling': 'COLING',
            'eccv': 'ECCV',
            'icassp': 'ICASSP',
            'icme': 'ICME',
            'icmr': 'ICMR',
            'icaps': 'ICAPS',
            'colt': 'COLT',
        }

    def _parse_authors(self, authors_data) -> List[str]:
        """Parse authors from various formats."""
        if not authors_data:
            return []

        if isinstance(authors_data, str):
            return [authors_data]

        if isinstance(authors_data, dict):
            author_list = authors_data.get('author', [])
            if isinstance(author_list, list):
                return [a.get('text', '') for a in author_list if a.get('text')]
            elif isinstance(author_list, dict):
                return [author_list.get('text', '')]
            return []

        if isinstance(authors_data, list):
            return [a.get('text', '') for a in authors_data if isinstance(a, dict) and a.get('text')]

        return []

    def _parse_ee(self, ee_data) -> Optional[str]:
        """Parse URL from ee field."""
        if not ee_data:
            return None

        if isinstance(ee_data, str):
            return ee_data

        if isinstance(ee_data, list):
            return ee_data[0] if ee_data else None

        return None

    def _load_paper_from_dict(self, paper_dict: dict, venue: str, year: int) -> Paper:
        """Load paper from dictionary."""
        title = paper_dict.get('title', '')

        # Parse authors
        authors = self._parse_authors(paper_dict.get('authors', []))

        # Parse URL
        url = self._parse_ee(paper_dict.get('ee'))

        # Parse DOI
        doi = paper_dict.get('doi', '')

        # Parse abstract
        abstract = paper_dict.get('abstract', '')

        # Parse DBLP key
        dblp_key = paper_dict.get('key', '')

        return Paper(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract if abstract else None,
            doi=doi if doi else None,
            url=url,
            dblp_key=dblp_key
        )

    def load_conference(self, conf_name: str, years: List[int] = None) -> ConferenceData:
        """
        Load conference data.

        Args:
            conf_name: Conference name (e.g., 'aaai', 'nips')
            years: List of years to load. If None, load all available.

        Returns:
            ConferenceData object
        """
        conf_lower = conf_name.lower()

        # Find the data directory containing conference files
        # Files are stored as: conf_a/aaai_2023.json, conf_a/acl_2023.json, etc.
        conf_dir = None

        # Look for directories containing conference data files
        for item in self.data_dir.iterdir():
            if item.is_dir() and 'conf' in item.name.lower():
                # Check if this directory contains files for this conference
                matching_files = list(item.glob(f"{conf_lower}_*.json"))
                if matching_files:
                    conf_dir = item
                    break

        if conf_dir is None:
            logger.warning(f"No data directory found for conference: {conf_name}")
            return ConferenceData(name=conf_lower, display_name=conf_name.upper())

        display_name = self.conf_name_mapping.get(conf_lower, conf_name.upper())

        logger.info(f"Loading {display_name} from {conf_dir}")

        papers = []

        # Find JSON files matching this conference
        json_files = sorted(conf_dir.glob(f"{conf_lower}_*.json"))

        for json_file in json_files:
            # Parse year from filename (e.g., aaai_2023.json)
            try:
                file_year = int(json_file.stem.split('_')[-1])
            except (ValueError, IndexError):
                logger.warning(f"Cannot parse year from filename: {json_file.name}")
                continue

            # Filter by year if specified
            if years and file_year not in years:
                continue

            # Load JSON file
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract papers
                papers_list = data.get('papers', [])
                for paper_dict in papers_list:
                    paper = self._load_paper_from_dict(paper_dict, display_name, file_year)
                    papers.append(paper)

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        # Determine year range
        if papers:
            year_range = (min(p.year for p in papers), max(p.year for p in papers))
        else:
            year_range = (0, 0)

        logger.info(f"Loaded {len(papers)} papers from {display_name} ({year_range[0]}-{year_range[1]})")

        return ConferenceData(
            name=conf_lower,
            display_name=display_name,
            papers=papers,
            year_range=year_range
        )

    def load_multiple_conferences(self, conf_names: List[str], years: List[int] = None) -> Dict[str, ConferenceData]:
        """
        Load multiple conferences.

        Args:
            conf_names: List of conference names
            years: List of years to load

        Returns:
            Dictionary mapping conference name to ConferenceData
        """
        result = {}
        for conf_name in conf_names:
            conf_data = self.load_conference(conf_name, years)
            result[conf_name] = conf_data

        return result

    def load_all_conferences(self, years: List[int] = None) -> Dict[str, ConferenceData]:
        """
        Load all available conferences.

        Args:
            years: List of years to load

        Returns:
            Dictionary mapping conference name to ConferenceData
        """
        conf_names = []

        # Find all conference directories
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('conf'):
                # Extract conference names from directory
                for json_file in item.glob("*.json"):
                    conf_name = json_file.stem.split('_')[0]
                    if conf_name not in conf_names:
                        conf_names.append(conf_name)

        logger.info(f"Found conferences: {conf_names}")
        return self.load_multiple_conferences(conf_names, years)

    def get_available_conferences(self) -> List[str]:
        """Get list of available conferences."""
        conferences = []

        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('conf'):
                for json_file in item.glob("*.json"):
                    conf_name = json_file.stem.split('_')[0]
                    if conf_name not in conferences:
                        conferences.append(conf_name)

        return sorted(conferences)

    def get_statistics(self) -> dict:
        """Get overall statistics."""
        all_confs = self.load_all_conferences()

        stats = {
            'total_conferences': len(all_confs),
            'total_papers': 0,
            'papers_with_abstract': 0,
            'conferences': {}
        }

        for conf_name, conf_data in all_confs.items():
            stats['total_papers'] += conf_data.paper_count
            stats['papers_with_abstract'] += conf_data.papers_with_abstract
            stats['conferences'][conf_name] = {
                'papers': conf_data.paper_count,
                'with_abstract': conf_data.papers_with_abstract,
                'year_range': conf_data.year_range
            }

        return stats


def load_papers_for_analysis(conf_names: List[str] = None, years: List[int] = None) -> List[Paper]:
    """
    Convenience function to load papers for analysis.

    Args:
        conf_names: Conference names to load. If None, load all.
        years: Years to load. If None, load all.

    Returns:
        List of Paper objects
    """
    loader = PaperDataLoader()

    if conf_names is None:
        conf_names = loader.get_available_conferences()

    all_papers = []

    for conf_name in conf_names:
        conf_data = loader.load_conference(conf_name, years)
        all_papers.extend(conf_data.papers)

    return all_papers


if __name__ == "__main__":
    # Test the loader
    loader = PaperDataLoader()

    print("Available conferences:", loader.get_available_conferences())

    # Load AAAI
    aaai_data = loader.load_conference('aaai', [2023, 2024])
    print(f"\nAAAI papers: {aaai_data.paper_count}")
    print(f"With abstract: {aaai_data.papers_with_abstract}")

    # Load statistics
    stats = loader.get_statistics()
    print(f"\nTotal papers: {stats['total_papers']}")
    print(f"Total with abstract: {stats['papers_with_abstract']}")
