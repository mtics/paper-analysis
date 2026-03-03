import pymannkendall as mk
from typing import Any, Dict, Literal


def mann_kendall_test(yearly_counts: Dict[int, int], alpha: float = 0.05) -> Dict[str, Any]:
    """
    对年度论文数量进行 Mann-Kendall 趋势检验

    Args:
        yearly_counts: {年份: 论文数量}
        alpha: 显著性水平阈值 (default: 0.05)

    Returns:
        {
            'trend': 'increasing' | 'decreasing' | 'no trend',
            'p_value': float,
            'significant': bool,
            'sens_slope': float,
            'tau': float
        }
    """
    if len(yearly_counts) < 3:
        return {
            'trend': 'no trend',
            'p_value': 1.0,
            'significant': False,
            'sens_slope': 0.0,
            'tau': 0.0
        }

    # 按年份排序
    years = sorted(yearly_counts.keys())
    counts = [yearly_counts[y] for y in years]

    # 执行 Mann-Kendall 检验
    result = mk.original_test(counts)

    return {
        'trend': result.trend,
        'p_value': result.p,
        'significant': result.p < alpha,
        'sens_slope': result.slope,
        'tau': result.Tau
    }


def normalize_yearly_counts(yearly_counts: Dict[int, int]) -> Dict[int, float]:
    """
    归一化年度论文数量为百分比（相对于当年总量的比例）

    用于跨年份对比时消除总量变化的影响
    """
    total = sum(yearly_counts.values())
    if total == 0:
        return {y: 0.0 for y in yearly_counts}
    return {y: count / total for y, count in yearly_counts.items()}


def calculate_growth_rate(
    yearly_counts: Dict[int, int], method: Literal['compound', 'simple'] = 'compound'
) -> float:
    """
    计算复合年增长率 (CAGR)

    Args:
        yearly_counts: {年份: 论文数量}
        method: 'compound' | 'simple'

    Returns:
        年均增长率 (如 0.15 表示 15%)

    Raises:
        ValueError: 如果 method 不是 'compound' 或 'simple'
    """
    if method not in ('compound', 'simple'):
        raise ValueError(f"Invalid method: {method}. Must be 'compound' or 'simple'.")

    years = sorted(yearly_counts.keys())
    if len(years) < 2:
        return 0.0

    start_year, end_year = years[0], years[-1]
    n_years = end_year - start_year
    if n_years <= 0:
        return 0.0

    start_count = yearly_counts[start_year]
    end_count = yearly_counts[end_year]

    if start_count <= 0:
        return 0.0

    if method == 'compound':
        cagr = (end_count / start_count) ** (1 / n_years) - 1
    else:
        cagr = (end_count - start_count) / (start_count * n_years)

    return cagr
