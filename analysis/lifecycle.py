# analysis/lifecycle.py

import logging
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def logistic(x: float, L: float, k: float, x0: float) -> float:
    """Logistic 函数用于 S 曲线拟合"""
    return L / (1 + np.exp(-k * (x - x0)))


class LifecycleAnalyzer:
    """领域生命周期分析 - S曲线拟合判断领域所处阶段"""

    def fit_scurve(
        self,
        yearly_counts: Dict[int, int]
    ) -> Dict:
        """
        对领域年度论文数拟合 logistic S 曲线

        Args:
            yearly_counts: {年份: 论文数量}

        Returns:
            {
                'stage': 'emerging' | 'growth' | 'mature' | 'declining',
                'L': float,  # 预估天花板
                'k': float,  # 增长速率
                'x0': float, # 拐点年份
                'r_squared': float, # 拟合优度
                'current_year': int,
                'current_count': int,
                'projected_peak': int or None
            }
        """
        if len(yearly_counts) < 3:
            return {'stage': 'unknown', 'error': 'insufficient data'}

        years = np.array(sorted(yearly_counts.keys()))
        counts = np.array([yearly_counts[y] for y in years])

        try:
            # 初始参数估计
            L0 = max(counts) * 2  # 预估天花板
            k0 = 0.5  # 增长速率
            x0 = np.median(years)  # 拐点

            popt, _ = curve_fit(
                logistic, years, counts,
                p0=[L0, k0, x0],
                bounds=([0, 0, min(years)], [max(counts) * 10, 10, max(years) + 5]),
                maxfev=5000
            )

            L, k, x0 = popt
            predicted = logistic(years, L, k, x0)
            ss_res = np.sum((counts - predicted) ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 判断阶段
            current_year = max(years)
            current_count = yearly_counts[current_year]

            # 增长阶段
            if k > 0 and current_year < x0:
                stage = 'growth'
            # 快速成熟
            elif current_count > L * 0.8:
                stage = 'mature'
            # 下降
            elif len(counts) > 2 and counts[-1] < counts[-2]:
                stage = 'declining'
            else:
                stage = 'mature'

            return {
                'stage': stage,
                'L': float(L),
                'k': float(k),
                'x0': float(x0),
                'r_squared': float(r_squared),
                'current_year': int(current_year),
                'current_count': int(current_count),
                'projected_peak': int(round(x0)) if x0 > max(years) else None
            }

        except Exception as e:
            logger.warning(f"S-curve fitting failed: {e}")
            # 回退到简单判断
            if len(counts) >= 2:
                growth = (counts[-1] - counts[0]) / (counts[0] + 1)
                return {
                    'stage': 'growth' if growth > 0.2 else 'mature',
                    'error': str(e)
                }
            return {'stage': 'unknown', 'error': str(e)}


class ResearcherStabilityAnalyzer:
    """研究者稳定性分析 - Jaccard 相似度"""

    def calculate_stability(
        self,
        papers: List,
        years: List[int]
    ) -> Dict:
        """
        计算领域作者稳定性 Jaccard 系数

        Returns:
            {
                'jaccard_by_year': {year_pair: float},
                'avg_jaccard': float,
                'stage': 'expert_dominant' | 'mixed' | 'inflow',
                'new_researcher_ratio': {year: float}
            }
        """
        # 按年份收集作者
        authors_by_year = defaultdict(set)
        for paper in papers:
            if paper.year in years:
                if hasattr(paper, 'authors') and paper.authors:
                    for author in paper.authors:
                        authors_by_year[paper.year].add(str(author))

        # 计算相邻年份 Jaccard
        jaccard_by_year = {}
        sorted_years = sorted(authors_by_year.keys())

        for i in range(len(sorted_years) - 1):
            y1, y2 = sorted_years[i], sorted_years[i + 1]
            set1 = authors_by_year[y1]
            set2 = authors_by_year[y2]

            if len(set1) > 0 and len(set2) > 0:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0
                jaccard_by_year[f"{y1}_{y2}"] = round(jaccard, 3)

        avg_jaccard = np.mean(list(jaccard_by_year.values())) if jaccard_by_year else 0

        # 判断阶段
        if avg_jaccard > 0.3:
            stage = 'expert_dominant'  # 专家主导
        elif avg_jaccard > 0.15:
            stage = 'mixed'  # 混合
        else:
            stage = 'inflow'  # 新涌入多

        # 计算每年新研究者比例
        new_ratio = {}
        all_authors = set()
        for year in sorted_years:
            year_authors = authors_by_year[year]
            new_count = len(year_authors - all_authors)
            new_ratio[year] = round(new_count / len(year_authors), 3) if year_authors else 0
            all_authors.update(year_authors)

        return {
            'jaccard_by_year': jaccard_by_year,
            'avg_jaccard': round(avg_jaccard, 3),
            'stage': stage,
            'new_researcher_ratio': new_ratio
        }
