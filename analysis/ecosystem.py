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


class ConferenceSimilarityMatrix:
    """会议相似度矩阵 - 量化 13 个 CCF-A 顶会的研究重心相似度"""

    def __init__(self, max_features: int = 5000):
        """
        Args:
            max_features: TF-IDF 最大特征数
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.max_features = max_features

    def compute_similarity(
        self,
        papers: List,
        conferences: List[str],
        years: List[int]
    ) -> Tuple["np.ndarray", List[str]]:
        """
        计算会议相似度矩阵

        Returns:
            (similarity_matrix, conference_names)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # 按会议聚合论文文本
        conf_texts = {}
        for conf in conferences:
            conf_papers = [p for p in papers if p.venue == conf and p.year in years]
            texts = []
            for p in conf_papers:
                text = p.title
                if p.has_abstract and p.abstract:
                    text += " " + p.abstract
                texts.append(text)
            conf_texts[conf] = " ".join(texts)

        # 计算 TF-IDF
        conf_names = [c for c in conferences if c in conf_texts and conf_texts[c]]
        texts = [conf_texts[c] for c in conf_names]

        if len(texts) < 2:
            return np.array([[1.0]]), conf_names

        tfidf = self.vectorizer.fit_transform(texts)

        # 计算余弦相似度
        similarity = cosine_similarity(tfidf)

        return similarity, conf_names

    def compare_eras(
        self,
        papers: List,
        conferences: List[str],
        era1_years: List[int],
        era2_years: List[int]
    ) -> Dict:
        """
        比较两个时代的相似度变化

        Args:
            papers: 论文列表
            conferences: 会议列表
            era1_years: 第一个时代年份
            era2_years: 第二个时代年份

        Returns:
            {
                "era1_years": era1_years,
                "era2_years": era2_years,
                "similarity_era1": np.ndarray,
                "similarity_era2": np.ndarray,
                "delta": np.ndarray,
                "conference_names": List[str]
            }
        """
        import numpy as np

        # 时代1
        era1_papers = [p for p in papers if p.year in era1_years]
        sim1, conf_names = self.compute_similarity(era1_papers, conferences, era1_years)

        # 时代2
        era2_papers = [p for p in papers if p.year in era2_years]
        sim2, _ = self.compute_similarity(era2_papers, conferences, era2_years)

        # 差值矩阵
        delta = sim2 - sim1

        return {
            "era1_years": era1_years,
            "era2_years": era2_years,
            "similarity_era1": sim1.tolist(),
            "similarity_era2": sim2.tolist(),
            "delta": delta.tolist(),
            "conference_names": conf_names
        }

    def find_converging_pairs(self, result: Dict, top_n: int = 10) -> List[Dict]:
        """找出融合最快的会议对"""
        delta = np.array(result["delta"])
        conf_names = result["conference_names"]

        pairs = []
        n = len(conf_names)
        for i in range(n):
            for j in range(i+1, n):
                pairs.append({
                    "conf1": conf_names[i],
                    "conf2": conf_names[j],
                    "delta": delta[i][j]
                })

        pairs.sort(key=lambda x: x["delta"], reverse=True)
        return pairs[:top_n]


class TechnologyDiffusion:
    """技术扩散路径 - 量化技术在各会议间的传播延迟"""

    def __init__(self, threshold: int = 10):
        """
        Args:
            threshold: 判断"达到该会议"的年度论文数量阈值
        """
        self.threshold = threshold

    def analyze_term(
        self,
        term: str,
        papers: List,
        conferences: List[str]
    ) -> Dict:
        """
        分析特定技术术语在各会议的扩散路径

        Args:
            term: 技术术语
            papers: 论文列表
            conferences: 会议列表

        Returns:
            {
                "term": str,
                "origin_conf": str,
                "origin_year": int,
                "diffusion_path": [
                    {"conf": str, "year": int, "delay": int},
                    ...
                ]
            }
        """
        # 统计每个会议每年的论文数
        term = term.lower()
        conf_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for paper in papers:
            if not paper.has_abstract or not paper.abstract:
                continue
            text = (paper.title + " " + paper.abstract).lower()
            if term in text:
                conf_year_counts[paper.venue][paper.year] += 1

        # 找出首次达到阈值的年份
        first_year_by_conf: Dict[str, int] = {}
        for conf in conferences:
            year_counts = conf_year_counts[conf]
            for year in sorted(year_counts.keys()):
                if year_counts[year] >= self.threshold:
                    first_year_by_conf[conf] = year
                    break

        if not first_year_by_conf:
            return {
                "term": term,
                "origin_conf": None,
                "origin_year": None,
                "diffusion_path": []
            }

        # 找到起源会议
        origin_conf = min(first_year_by_conf, key=first_year_by_conf.get)
        origin_year = first_year_by_conf[origin_conf]

        # 构建扩散路径
        diffusion_path = []
        for conf in sorted(first_year_by_conf.keys(), key=lambda c: first_year_by_conf[c]):
            diffusion_path.append({
                "conf": conf,
                "year": first_year_by_conf[conf],
                "delay": first_year_by_conf[conf] - origin_year
            })

        return {
            "term": term,
            "origin_conf": origin_conf,
            "origin_year": origin_year,
            "diffusion_path": diffusion_path
        }
