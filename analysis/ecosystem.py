# analysis/ecosystem.py

import logging
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from analysis.data_loader import Paper

logger = logging.getLogger(__name__)


class VocabularyTimeline:
    """词汇时间轴 - 追踪技术词汇从"首次出现"到"全面扩散"的演变"""

    def __init__(self, min_count: int = 10, min_total_count: int = 50):
        """
        Args:
            min_count: 最小年度词频阈值，用于判断"集中出现"
            min_total_count: 最小总词频阈值，用于过滤太低频的词
        """
        self.min_count = min_count
        self.min_total_count = min_total_count

    def analyze(
        self,
        papers: List["Paper"],
        top_n: int = 1000
    ) -> pd.DataFrame:
        """
        分析词汇时间轴

        Args:
            papers: 论文列表
            top_n: 分析前 N 个高频词

        Returns:
            DataFrame with columns: phrase, first_year, first_cluster_year,
            peak_year, spread_speed, trajectory, origin_conf
        """
        # 1. 统计每个词在每个会议每年的出现次数
        phrase_by_conf_year = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for paper in papers:
            if not paper.has_abstract:
                continue
            text = (paper.title + " " + (paper.abstract or "")).lower()
            year = paper.year
            conf = paper.venue

            if conf is None:
                continue

            # 简单分词（后续可用 NgramExtractor 增强）
            words = text.split()
            for word in words:
                if len(word) > 3:  # 过滤短词
                    phrase_by_conf_year[word][conf][year] += 1

        # 2. 构建时间轴数据
        results = []
        for phrase, confs in phrase_by_conf_year.items():
            yearly_total = defaultdict(int)
            for conf, years in confs.items():
                for year, count in years.items():
                    yearly_total[year] += count

            if sum(yearly_total.values()) < self.min_total_count:
                continue  # 跳过太低频的词

            # 计算各项指标
            years = sorted(yearly_total.keys())
            first_year = min(years)

            # 首次集中出现（>= min_count 篇）
            first_cluster_year = None
            for y in years:
                if yearly_total[y] >= self.min_count:
                    first_cluster_year = y
                    break

            # 峰值年份
            peak_year = max(yearly_total, key=yearly_total.get)

            # 扩散速度：从 10 篇到 100 篇（或峰值）需要的年数
            spread_speed = None
            if yearly_total.get(peak_year, 0) >= 100:
                y10 = None
                y100 = None
                for y in years:
                    if yearly_total[y] >= 10 and y10 is None:
                        y10 = y
                    if yearly_total[y] >= 100 and y100 is None:
                        y100 = y
                    if y10 and y100:
                        break
                if y10 and y100:
                    spread_speed = y100 - y10

            # 轨迹判断
            recent_years = years[-3:] if len(years) >= 3 else years
            recent_avg = np.mean([yearly_total[y] for y in recent_years])
            early_avg = np.mean([yearly_total[y] for y in years[:3]])

            if recent_avg > early_avg * 1.5:
                trajectory = "rising"
            elif recent_avg < early_avg * 0.5:
                trajectory = "declining"
            elif yearly_total[peak_year] - yearly_total[first_year] > 50:
                trajectory = "peaked"
            else:
                trajectory = "niche"

            # 起源会议
            origin_conf = None
            for conf, years_data in confs.items():
                if sum(years_data.values()) >= self.min_count:
                    origin_conf = conf
                    break

            results.append({
                "phrase": phrase,
                "first_year": first_year,
                "first_cluster_year": first_cluster_year,
                "peak_year": peak_year,
                "spread_speed": spread_speed,
                "trajectory": trajectory,
                "origin_conf": origin_conf,
                "yearly_counts": dict(yearly_total)
            })

        df = pd.DataFrame(results)
        if not df.empty and "spread_speed" in df.columns:
            df = df.sort_values("spread_speed")
        return df

    def get_paradigm_shifts(self, df: pd.DataFrame, top_n: int = 20) -> List[Dict]:
        """返回扩散最快的技术词汇——即范式跃迁标志"""
        if "spread_speed" not in df.columns or df.empty:
            return []

        valid = df[df["spread_speed"].notna()]
        if valid.empty:
            return []

        valid = valid.sort_values("spread_speed").head(top_n)

        # 返回时排除 yearly_counts（太长）
        result = []
        for _, row in valid.iterrows():
            d = row.to_dict()
            d.pop("yearly_counts", None)
            result.append(d)

        return result
