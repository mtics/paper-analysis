#!/usr/bin/env python3
"""
Trend analysis module.
Analyze trends in papers over time.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

from analysis.features.trends.stats_utils import mann_kendall_test
from analysis.data.vocabulary import STOPWORDS as VOCABULARY_STOPWORDS, SYNONYMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrendReport:
    """Trend analysis report."""
    conference: str
    year_range: Tuple[int, int]
    total_papers: int
    papers_with_abstract: int
    yearly_counts: Dict[int, int]
    domain_distribution: Dict[str, int]
    top_keywords: Dict[str, int]
    emerging_keywords: List[Tuple[str, float]]


class TrendAnalyzer:
    """Analyze trends in conference papers."""

    def __init__(self):
        """Initialize trend analyzer."""
        pass

    def analyze_yearly_trends(self, papers: List) -> Dict:
        """
        Analyze yearly trends.

        Args:
            papers: List of Paper objects

        Returns:
            Yearly trend data
        """
        yearly_counts = Counter()
        yearly_with_abstract = Counter()
        yearly_authors = defaultdict(set)

        for paper in papers:
            yearly_counts[paper.year] += 1
            if paper.has_abstract:
                yearly_with_abstract[paper.year] += 1
            for author in paper.authors:
                yearly_authors[paper.year].add(author)

        # Sort by year
        years = sorted(yearly_counts.keys())

        return {
            'years': years,
            'paper_counts': [yearly_counts[y] for y in years],
            'papers_with_abstract': [yearly_with_abstract[y] for y in years],
            'unique_authors': [len(yearly_authors[y]) for y in years],
            'avg_papers_per_author': [
                yearly_counts[y] / len(yearly_authors[y]) if yearly_authors[y] else 0
                for y in years
            ]
        }

    def analyze_keyword_trends(self, papers: List, top_n: int = 50) -> Dict:
        """
        Analyze keyword trends over time.

        Args:
            papers: List of Paper objects
            top_n: Number of top keywords to track

        Returns:
            Keyword trend data
        """
        # Get all years
        years = sorted(set(p.year for p in papers))

        # Count keywords per year
        keyword_by_year = defaultdict(lambda: defaultdict(int))
        keyword_total = Counter()

        # Use vocabulary module for stopwords
        stopwords = VOCABULARY_STOPWORDS

        for paper in papers:
            text = paper.title
            if paper.has_abstract:
                text += " " + paper.abstract

            words = text.lower().split()
            words = [w for w in words if len(w) > 2 and w not in stopwords]

            for word in set(words):  # Use set to count unique words per paper
                keyword_by_year[word][paper.year] += 1
                keyword_total[word] += 1

        # Get top keywords
        top_keywords = [kw for kw, _ in keyword_total.most_common(top_n)]

        # Build trend data for top keywords
        keyword_trends = {}
        for keyword in top_keywords:
            keyword_trends[keyword] = [
                {
                    'year': year,
                    'count': keyword_by_year[keyword].get(year, 0)
                }
                for year in years
            ]

        # Detect emerging keywords using Mann-Kendall trend test
        growth_rates = []
        for keyword in top_keywords:
            if len(keyword_by_year[keyword]) >= 3:  # Mann-Kendall needs at least 3 data points
                # Convert to {year: count} format
                yearly_counts = {int(y): c for y, c in keyword_by_year[keyword].items()}

                # Mann-Kendall trend test
                mk_result = mann_kendall_test(yearly_counts)

                if mk_result['significant'] and mk_result['trend'] == 'increasing':
                    growth_rates.append({
                        'word': keyword,
                        'growth_rate': mk_result['sens_slope'],  # Use Sen's slope
                        'p_value': mk_result['p_value'],
                        'trend': mk_result['trend']
                    })

        # Sort by growth rate
        growth_rates.sort(key=lambda x: x['growth_rate'], reverse=True)

        # Format for return
        emerging = [(item['word'], item['growth_rate']) for item in growth_rates]

        return {
            'years': years,
            'top_keywords': top_keywords[:50],
            'keyword_trends': keyword_trends,
            'emerging_keywords': emerging[:20],
            'keyword_total': dict(keyword_total.most_common(100))
        }

    def analyze_author_trends(self, papers: List, top_n: int = 20) -> Dict:
        """
        Analyze author trends.

        Args:
            papers: List of Paper objects
            top_n: Number of top authors to track

        Returns:
            Author trend data
        """
        author_year_counts = defaultdict(lambda: defaultdict(int))
        author_total = Counter()

        for paper in papers:
            for author in paper.authors:
                author_year_counts[author][paper.year] += 1
                author_total[author] += 1

        # Get top authors
        top_authors = [a for a, _ in author_total.most_common(top_n)]

        # Build trend data
        years = sorted(set(p.year for p in papers))

        author_trends = {}
        for author in top_authors:
            author_trends[author] = [
                {
                    'year': year,
                    'count': author_year_counts[author].get(year, 0)
                }
                for year in years
            ]

        # Most prolific authors
        prolific_authors = author_total.most_common(30)

        return {
            'years': years,
            'top_authors': top_authors,
            'author_trends': author_trends,
            'prolific_authors': prolific_authors
        }

    def analyze_venue_trends(self, papers: List) -> Dict:
        """
        Analyze venue distribution trends.

        Args:
            papers: List of Paper objects

        Returns:
            Venue trend data
        """
        venue_year_counts = defaultdict(lambda: defaultdict(int))

        for paper in papers:
            venue_year_counts[paper.venue][paper.year] += 1

        years = sorted(set(p.year for p in papers))

        # Get venue totals
        venue_totals = Counter()
        for paper in papers:
            venue_totals[paper.venue] += 1

        top_venues = [v for v, _ in venue_totals.most_common(10)]

        venue_trends = {}
        for venue in top_venues:
            venue_trends[venue] = [
                {
                    'year': year,
                    'count': venue_year_counts[venue].get(year, 0)
                }
                for year in years
            ]

        return {
            'years': years,
            'top_venues': top_venues,
            'venue_trends': venue_trends,
            'venue_totals': dict(venue_totals.most_common(20))
        }

    def analyze_abstract_coverage(self, papers: List) -> Dict:
        """
        Analyze abstract coverage over time.

        Args:
            papers: List of Paper objects

        Returns:
            Abstract coverage data
        """
        yearly_total = Counter()
        yearly_with_abstract = Counter()

        for paper in papers:
            yearly_total[paper.year] += 1
            if paper.has_abstract:
                yearly_with_abstract[paper.year] += 1

        years = sorted(yearly_total.keys())

        coverage = []
        for year in years:
            total = yearly_total[year]
            with_abstract = yearly_with_abstract[year]
            coverage_rate = with_abstract / total if total > 0 else 0

            coverage.append({
                'year': year,
                'total': total,
                'with_abstract': with_abstract,
                'coverage_rate': coverage_rate
            })

        return {
            'years': years,
            'coverage': coverage,
            'overall_rate': sum(yearly_with_abstract.values()) / sum(yearly_total.values())
            if yearly_total else 0
        }


class ComparativeAnalyzer:
    """Compare trends across conferences."""

    def compare_conferences(self, conf_data_dict: Dict[str, List]) -> Dict:
        """
        Compare trends across multiple conferences.

        Args:
            conf_data_dict: Dictionary mapping conference name to papers

        Returns:
            Comparison data
        """
        comparison = {
            'conferences': list(conf_data_dict.keys()),
            'total_papers': {},
            'yearly_counts': {},
            'avg_papers_per_year': {},
            'top_keywords': {}
        }

        for conf_name, papers in conf_data_dict.items():
            # Total papers
            comparison['total_papers'][conf_name] = len(papers)

            # Yearly counts
            yearly = Counter(p.year for p in papers)
            comparison['yearly_counts'][conf_name] = dict(sorted(yearly.items()))

            # Average papers per year
            years = set(p.year for p in papers)
            comparison['avg_papers_per_year'][conf_name] = len(papers) / len(years) if years else 0

            # Top keywords (simplified)
            text = " ".join([
                p.title + (" " + p.abstract if p.has_abstract else "")
                for p in papers
            ])
            words = text.lower().split()
            # Use vocabulary module for stopwords
            words = [w for w in words if len(w) > 2 and w not in VOCABULARY_STOPWORDS]
            word_counts = Counter(words)
            comparison['top_keywords'][conf_name] = dict(word_counts.most_common(20))

        return comparison


def generate_trend_report(papers: List, conference_name: str) -> TrendReport:
    """
    Generate a comprehensive trend report.

    Args:
        papers: List of papers
        conference_name: Name of conference

    Returns:
        TrendReport object
    """
    analyzer = TrendAnalyzer()

    # Basic stats
    yearly_counts = Counter(p.year for p in papers)
    papers_with_abstract = sum(1 for p in papers if p.has_abstract)

    # Get keyword trends
    keyword_data = analyzer.analyze_keyword_trends(papers)

    # Simple domain distribution
    domain_counts = Counter()
    for paper in papers:
        text = paper.title + (" " + paper.abstract if paper.has_abstract else "")
        text_lower = text.lower()

        if any(kw in text_lower for kw in ['neural network', 'deep learning', 'machine learning']):
            domain_counts['Machine Learning'] += 1
        elif any(kw in text_lower for kw in ['image', 'vision', 'object detection', 'segmentation']):
            domain_counts['Computer Vision'] += 1
        elif any(kw in text_lower for kw in ['natural language', 'nlp', 'language model', 'text', 'translation']):
            domain_counts['NLP'] += 1
        elif any(kw in text_lower for kw in ['database', 'query', 'sql', 'index']):
            domain_counts['Database'] += 1
        elif any(kw in text_lower for kw in ['security', 'privacy', 'attack', 'cryptography']):
            domain_counts['Security'] += 1
        else:
            domain_counts['Other'] += 1

    return TrendReport(
        conference=conference_name,
        year_range=(min(yearly_counts.keys()), max(yearly_counts.keys())) if yearly_counts else (0, 0),
        total_papers=len(papers),
        papers_with_abstract=papers_with_abstract,
        yearly_counts=dict(yearly_counts),
        domain_distribution=dict(domain_counts),
        top_keywords=keyword_data.get('keyword_total', {}),
        emerging_keywords=keyword_data.get('emerging_keywords', []),
    )


if __name__ == "__main__":
    # Test trend analysis
    from analysis.core.data_loader import PaperDataLoader

    loader = PaperDataLoader()

    # Load AAAI papers
    aaai_data = loader.load_conference('aaai', [2022, 2023, 2024])
    papers = aaai_data.papers

    print(f"Loaded {len(papers)} papers from AAAI")

    # Analyze trends
    analyzer = TrendAnalyzer()

    yearly = analyzer.analyze_yearly_trends(papers)
    print("\nYearly trends:")
    for i, year in enumerate(yearly['years']):
        print(f"  {year}: {yearly['paper_counts'][i]} papers")

    keywords = analyzer.analyze_keyword_trends(papers)
    print("\nTop 10 keywords:", list(keywords['keyword_total'].keys())[:10])
    print("\nEmerging keywords:", keywords['emerging_keywords'][:10])
