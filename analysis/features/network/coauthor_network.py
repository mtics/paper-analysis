# analysis/network_analysis.py

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class CoauthorNetworkAnalyzer:
    """共作者网络分析 - 识别桥接者和社区结构"""

    def build_graph(
        self,
        papers: List,
        year: Optional[int] = None
    ) -> nx.Graph:
        """
        构建共作者网络

        Args:
            papers: 论文列表
            year: 可选，只包含特定年份的论文

        Returns:
            NetworkX 无向图
        """
        G = nx.Graph()

        # 筛选论文
        if year:
            papers = [p for p in papers if p.year == year]

        # 添加边
        for paper in papers:
            authors = paper.authors if hasattr(paper, 'authors') else []
            if len(authors) < 2:
                continue

            # 完全图连接所有作者
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    a1_str = str(a1)
                    a2_str = str(a2)
                    if G.has_edge(a1_str, a2_str):
                        G[a1_str][a2_str]['weight'] += 1
                    else:
                        G.add_edge(a1_str, a2_str, weight=1)

        return G

    def analyze_evolution(
        self,
        papers: List,
        years: List[int]
    ) -> pd.DataFrame:
        """
        逐年计算网络结构指标

        Returns:
            DataFrame with columns: year, num_nodes, num_edges,
            avg_path_length, clustering_coefficient, network_density
        """
        results = []

        for year in years:
            G = self.build_graph(papers, year)

            if G.number_of_nodes() < 2:
                results.append({
                    "year": year,
                    "num_nodes": 0,
                    "num_edges": 0,
                    "avg_path_length": None,
                    "clustering_coefficient": None,
                    "network_density": None
                })
                continue

            try:
                avg_path = nx.average_shortest_path_length(G)
            except (nx.NetworkXError, nx.NetworkXException):
                avg_path = None

            results.append({
                "year": year,
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "avg_path_length": avg_path,
                "clustering_coefficient": nx.average_clustering(G),
                "network_density": nx.density(G)
            })

        return pd.DataFrame(results)

    def find_bridge_researchers(
        self,
        G: nx.Graph,
        top_n: int = 20
    ) -> List[Dict]:
        """识别高 betweenness centrality 的桥接者"""
        if G.number_of_nodes() < 2:
            return []

        betweenness = nx.betweenness_centrality(G)
        top_researchers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [{"name": name, "betweenness": score} for name, score in top_researchers]

    def find_cross_venue_researchers(
        self,
        papers: List,
        venue_pairs: List[Tuple[str, str]],
        min_papers: int = 3
    ) -> List[Dict]:
        """识别跨会议活跃的研究者"""
        author_venues = defaultdict(set)

        for paper in papers:
            authors = paper.authors if hasattr(paper, 'authors') else []
            for author in authors:
                author_venues[str(author)].add(paper.venue)

        results = []
        for author, venues in author_venues.items():
            for v1, v2 in venue_pairs:
                if v1 in venues and v2 in venues:
                    results.append({
                        "name": author,
                        "venues": list(venues),
                        "is_bridge": True
                    })

        return sorted(results, key=lambda x: len(x["venues"]), reverse=True)[:min_papers * 10]
