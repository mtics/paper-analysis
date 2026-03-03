# Trend analysis features
from analysis.features.trends.trend_analyzer import TrendReport, TrendAnalyzer, ComparativeAnalyzer, generate_trend_report
from analysis.features.trends.stats_utils import mann_kendall_test, normalize_yearly_counts, calculate_growth_rate

__all__ = [
    'TrendReport', 'TrendAnalyzer', 'ComparativeAnalyzer', 'generate_trend_report',
    'mann_kendall_test', 'normalize_yearly_counts', 'calculate_growth_rate'
]
