# tests/test_lifecycle.py

import pytest
from analysis.lifecycle import LifecycleAnalyzer, ResearcherStabilityAnalyzer, logistic
from analysis.data_loader import Paper
import numpy as np


class TestLifecycleAnalyzer:

    def test_fit_scurve_growth(self):
        yearly = {2018: 10, 2019: 50, 2020: 150, 2021: 300, 2022: 400}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert 'stage' in result
        assert result['stage'] in ['growth', 'mature', 'declining', 'emerging']

    def test_fit_scurve_insufficient(self):
        yearly = {2022: 100}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert result['stage'] == 'unknown'
        assert 'error' in result

    def test_fit_scurve_declining(self):
        yearly = {2018: 400, 2019: 300, 2020: 200, 2021: 150, 2022: 100}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert 'stage' in result
        assert result['stage'] in ['growth', 'mature', 'declining', 'emerging']

    def test_fit_scurve_mature(self):
        yearly = {2018: 100, 2019: 120, 2020: 115, 2021: 125, 2022: 118}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert 'stage' in result

    def test_fit_scurve_returns_params(self):
        yearly = {2018: 10, 2019: 50, 2020: 150, 2021: 300, 2022: 400}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert 'L' in result
        assert 'k' in result
        assert 'x0' in result
        assert 'r_squared' in result
        assert 'current_year' in result
        assert 'current_count' in result


class TestResearcherStabilityAnalyzer:

    def test_calculate_stability(self):
        # Create mock Paper objects
        papers = [
            Paper(title="Paper 1", authors=["Alice", "Bob"], year=2020, venue="CVPR"),
            Paper(title="Paper 2", authors=["Alice", "Charlie"], year=2020, venue="CVPR"),
            Paper(title="Paper 3", authors=["Alice", "Bob"], year=2021, venue="CVPR"),
            Paper(title="Paper 4", authors=["Dave", "Eve"], year=2021, venue="CVPR"),
            Paper(title="Paper 5", authors=["Frank", "Alice"], year=2022, venue="CVPR"),
        ]
        years = [2020, 2021, 2022]

        analyzer = ResearcherStabilityAnalyzer()
        result = analyzer.calculate_stability(papers, years)

        assert 'jaccard_by_year' in result
        assert 'avg_jaccard' in result
        assert 'stage' in result
        assert 'new_researcher_ratio' in result
        assert result['stage'] in ['expert_dominant', 'mixed', 'inflow']

    def test_calculate_stability_empty_papers(self):
        papers = []
        years = [2020, 2021, 2022]

        analyzer = ResearcherStabilityAnalyzer()
        result = analyzer.calculate_stability(papers, years)

        assert result['avg_jaccard'] == 0
        assert 'jaccard_by_year' in result

    def test_calculate_stability_single_year(self):
        papers = [
            Paper(title="Paper 1", authors=["Alice", "Bob"], year=2020, venue="CVPR"),
        ]
        years = [2020]

        analyzer = ResearcherStabilityAnalyzer()
        result = analyzer.calculate_stability(papers, years)

        assert 'jaccard_by_year' in result
        assert len(result['jaccard_by_year']) == 0  # No pairs to compare

    def test_calculate_stability_no_authors(self):
        papers = [
            Paper(title="Paper 1", authors=[], year=2020, venue="CVPR"),
            Paper(title="Paper 2", authors=[], year=2021, venue="CVPR"),
        ]
        years = [2020, 2021]

        analyzer = ResearcherStabilityAnalyzer()
        result = analyzer.calculate_stability(papers, years)

        assert result['avg_jaccard'] == 0


class TestLogistic:

    def test_logistic_function(self):
        result = logistic(2020, L=1000, k=1, x0=2020)
        assert result == 500.0  # At x0, logistic returns L/2

    def test_logistic_function_asymptote_positive(self):
        # As x -> infinity, logistic -> L
        result = logistic(3000, L=1000, k=1, x0=2020)
        assert result > 900  # Very close to L

    def test_logistic_function_asymptote_negative(self):
        # As x -> -infinity, logistic -> 0
        result = logistic(1000, L=1000, k=1, x0=2020)
        assert result < 0.001  # Very close to 0

    def test_logistic_function_growth(self):
        # Test that logistic increases with x when k > 0
        result1 = logistic(2018, L=1000, k=0.5, x0=2020)
        result2 = logistic(2022, L=1000, k=0.5, x0=2020)
        assert result1 < result2

    def test_logistic_function_k_negative(self):
        # Test with negative k (declining)
        result1 = logistic(2018, L=1000, k=-0.5, x0=2020)
        result2 = logistic(2022, L=1000, k=-0.5, x0=2020)
        assert result1 > result2
