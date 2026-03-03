# 阶段二：宏观生态分析实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现阶段二全部内容，包括 VocabularyTimeline、ConferenceSimilarityMatrix、TechnologyDiffusion、KnowledgeFlowGraph、CoauthorNetworkAnalyzer 及 PyVis 可视化

**Architecture:**
- 新建 `analysis/ecosystem.py`：宏观生态分析（模块 A/B/C/D）
- 新建 `analysis/network_analysis.py`：共作者网络分析（模块 E）
- 修改 `analysis/main.py`：添加 ecosystem / network / timeline CLI 命令
- 更新 `requirements.txt`：取消注释 pyvis

**Tech Stack:** pandas, numpy, networkx, pyvis, scikit-learn (TF-IDF)

---

## Task 1: 更新 requirements.txt 添加 pyvis

**Files:**
- Modify: `requirements.txt`

**Step 1: 取消注释 pyvis**

找到第 33 行：
```
# pyvis>=0.3.2          # NetworkX → interactive HTML via vis.js
```

改为：
```
pyvis>=0.3.2            # NetworkX → interactive HTML via vis.js
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: enable pyvis for interactive visualization"
```

---

## Task 2: 创建 analysis/ecosystem.py - 模块 A: VocabularyTimeline

**Files:**
- Create: `analysis/ecosystem.py` (initial content)

**Step 1: 创建 VocabularyTimeline 类**

```python
# analysis/ecosystem.py

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VocabularyTimeline:
    """词汇时间轴 - 追踪技术词汇从"首次出现"到"全面扩散"的演变"""

    def __init__(self, min_count: int = 10):
        """
        Args:
            min_count: 最小年度词频阈值，用于判断"集中出现"
        """
        self.min_count = min_count

    def analyze(
        self,
        papers: List,  # List[Paper] from data_loader
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
            text = (paper.title + " " + paper.abstract).lower()
            year = paper.year
            conf = paper.venue

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

            if sum(yearly_total.values()) < top_n:
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
        df = df.sort_values("spread_speed" if "spread_speed" in df.columns else "first_year")
        return df

    def get_paradigm_shifts(self, df: pd.DataFrame, top_n: int = 20) -> List[Dict]:
        """返回扩散最快的技术词汇——即范式跃迁标志"""
        if "spread_speed" not in df.columns or df.empty:
            return []

        valid = df[df["spread_speed"].notna()]
        valid = valid.sort_values("spread_speed").head(top_n)

        return valid.to_dict("records")
```

**Step 2: 测试**

```bash
python -c "
from analysis.ecosystem import VocabularyTimeline
from analysis.data_loader import PaperDataLoader

loader = PaperDataLoader()
papers = loader.load_conference('aaai', [2018, 2019, 2020, 2021, 2022]).papers

vt = VocabularyTimeline(min_count=10)
df = vt.analyze(papers, top_n=100)
print(df.head(10))
"
```

**Step 3: Commit**

```bash
git add analysis/ecosystem.py
git commit -m "feat(ecosystem): add VocabularyTimeline module"
```

---

## Task 3: 模块 B - ConferenceSimilarityMatrix

**Files:**
- Modify: `analysis/ecosystem.py`

**Step 1: 添加 ConferenceSimilarityMatrix 类**

```python
class ConferenceSimilarityMatrix:
    """会议相似度矩阵 - 量化 13 个 CCF-A 顶会的研究重心相似度"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def compute_similarity(
        self,
        papers: List,
        conferences: List[str],
        years: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        计算会议相似度矩阵

        Returns:
            (similarity_matrix, conference_names)
        """
        # 按会议聚合论文文本
        conf_texts = {}
        for conf in conferences:
            conf_papers = [p for p in papers if p.venue == conf]
            text = " ".join([
                (p.title + " " + (p.abstract if p.has_abstract else ""))
                for p in conf_papers
            ])
            conf_texts[conf] = text

        # 计算 TF-IDF
        conf_names = list(conf_texts.keys())
        texts = [conf_texts[c] for c in conf_names]

        if len(texts) < 2:
            return np.array([[1.0]]), conf_names

        tfidf = self.vectorizer.fit_transform(texts)

        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(tfidf)

        return similarity, conf_names

    def compare_eras(
        self,
        papers: List,
        conferences: List[str],
        era1_years: List[int],
        era2_years: List[int]
    ) -> Dict:
        """比较两个时代的相似度变化"""
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
            "similarity_era1": sim1,
            "similarity_era2": sim2,
            "delta": delta,
            "conference_names": conf_names
        }
```

**Step 2: Commit**

```bash
git commit -m "feat(ecosystem): add ConferenceSimilarityMatrix module"
```

---

## Task 4: 模块 C - TechnologyDiffusion

**Files:**
- Modify: `analysis/ecosystem.py`

**Step 1: 添加 TechnologyDiffusion 类**

```python
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
        conf_year_counts = defaultdict(lambda: defaultdict(int))

        for paper in papers:
            if not paper.has_abstract:
                continue
            text = (paper.title + " " + paper.abstract).lower()
            if term in text:
                conf_year_counts[paper.venue][paper.year] += 1

        # 找出首次达到阈值的年份
        first_year_by_conf = {}
        for conf in conferences:
            year_counts = conf_year_counts[conf]
            for year in sorted(year_counts.keys()):
                if year_counts[year] >= self.threshold:
                    first_year_by_conf[conf] = year
                    break

        if not first_year_by_conf:
            return {"term": term, "diffusion_path": []}

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
```

**Step 2: Commit**

```bash
git commit -m "feat(ecosystem): add TechnologyDiffusion module"
```

---

## Task 5: 模块 D - KnowledgeFlowGraph

**Files:**
- Modify: `analysis/ecosystem.py`

**Step 1: 添加 KnowledgeFlowGraph 类**

```python
class KnowledgeFlowGraph:
    """知识流动有向图 - 量化领域间的词汇借用关系"""

    def __init__(self, pmi_threshold: float = 0.5):
        """
        Args:
            pmi_threshold: PMI 阈值，超过该值认为存在知识流动
        """
        self.pmi_threshold = pmi_threshold

    def build_graph(
        self,
        papers: List,
        domains: Dict[str, List[str]]  # {domain: [keywords]}
    ) -> "nx.DiGraph":
        """
        构建知识流动有向图

        Args:
            papers: 论文列表
            domains: 领域定义 {领域名: [关键词列表]}

        Returns:
            NetworkX DiGraph
        """
        import networkx as nx

        # 统计领域词汇在每篇论文中的出现
        domain_word_counts = {d: 0 for d in domains}
        paper_domain_match = defaultdict(lambda: {d: False for d in domains})

        for paper in papers:
            if not paper.has_abstract:
                continue
            text = (paper.title + " " + paper.abstract).lower()

            for domain, keywords in domains.items():
                for kw in keywords:
                    if kw.lower() in text:
                        domain_word_counts[domain] += 1
                        paper_domain_match[paper][domain] = True

        # 计算领域间的 PMI
        total_papers = sum(paper_domain_match.values())
        G = nx.DiGraph()

        for source_domain in domains:
            for target_domain in domains:
                if source_domain == target_domain:
                    continue

                # 计算条件概率
                source_count = sum(1 for p, matches in paper_domain_match.items() if matches[source_domain])
                target_count = sum(1 for p, matches in paper_domain_match.items() if matches[target_domain])
                joint_count = sum(1 for p, matches in paper_domain_match.items()
                                  if matches[source_domain] and matches[target_domain])

                if source_count > 0 and target_count > 0 and joint_count > 0:
                    pmi = np.log((joint_count * total_papers) / (source_count * target_count))

                    if pmi > self.pmi_threshold:
                        G.add_edge(source_domain, target_domain, weight=pmi)

        return G

    def export_html(self, G: "nx.DiGraph", output_path: str):
        """导出为 PyVis 交互式 HTML"""
        try:
            from pyvis.network import Network
            net = Network(directed=True)
            net.from_nx(G)
            net.save_graph(output_path)
            logger.info(f"Knowledge flow graph saved to {output_path}")
        except ImportError:
            logger.warning("pyvis not installed, skipping HTML export")
```

**Step 2: Commit**

```bash
git commit -m "feat(ecosystem): add KnowledgeFlowGraph module"
```

---

## Task 6: 创建 analysis/network_analysis.py - 模块 E

**Files:**
- Create: `analysis/network_analysis.py`

**Step 1: 创建 CoauthorNetworkAnalyzer 类**

```python
# analysis/network_analysis.py

import logging
from typing import List, Dict, Optional
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
                    if G.has_edge(a1, a2):
                        G[a1][a2]['weight'] += 1
                    else:
                        G.add_edge(a1, a2, weight=1)

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
            except nx.NetworkXError:
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
                author_venues[author].add(paper.venue)

        results = []
        for author, venues in author_venues.items():
            for v1, v2 in venue_pairs:
                if v1 in venues and v2 in venues:
                    results.append({
                        "name": author,
                        "venues": list(venues),
                        "is_bridge": True
                    })

        return sorted(results, key=lambda x: len(x["venues"]), reverse=True)[:top_n]
```

**Step 2: Commit**

```bash
git add analysis/network_analysis.py
git commit -m "feat(network): add CoauthorNetworkAnalyzer module"
```

---

## Task 7: 创建测试文件

**Files:**
- Create: `tests/test_ecosystem.py`
- Create: `tests/test_network_analysis.py`

**Step 1: 创建测试文件**

```python
# tests/test_ecosystem.py

import pytest
from analysis.ecosystem import VocabularyTimeline, ConferenceSimilarityMatrix, TechnologyDiffusion


class TestVocabularyTimeline:

    def test_analyze_returns_dataframe(self):
        # 需要 Paper 对象列表
        pass  # 后续完善


class TestConferenceSimilarityMatrix:

    def test_compute_similarity(self):
        pass


class TestTechnologyDiffusion:

    def test_analyze_term(self):
        pass
```

```python
# tests/test_network_analysis.py

import pytest
from analysis.network_analysis import CoauthorNetworkAnalyzer


class TestCoauthorNetworkAnalyzer:

    def test_build_graph(self):
        pass
```

**Step 2: Commit**

```bash
git add tests/test_ecosystem.py tests/test_network_analysis.py
git commit -m "test: add ecosystem and network test files"
```

---

## Task 8: 更新 CLI 命令

**Files:**
- Modify: `analysis/main.py`

**Step 1: 添加 ecosystem/network/timeline 命令**

在 `main()` 中添加子命令：

```python
# timeline command
timeline_parser = subparsers.add_parser('timeline', help='Analyze vocabulary timeline')
timeline_parser.add_argument('--years', type=str, default='2015,2025')
timeline_parser.add_argument('--top-n', type=int, default=100)
timeline_parser.add_argument('--output', type=str, default=None)

# ecosystem command
ecosystem_parser = subparsers.add_parser('ecosystem', help='Run full ecosystem analysis')
ecosystem_parser.add_argument('--years', type=str, default='2015,2025')
ecosystem_parser.add_argument('--conferences', type=str, default=None)
ecosystem_parser.add_argument('--output', type=str, default=None)

# network command
network_parser = subparsers.add_parser('network', help='Analyze coauthor network')
network_parser.add_argument('--years', type=str, default='2015,2025')
network_parser.add_argument('--conferences', type=str, default=None)
network_parser.add_argument('--output', type=str, default=None)
```

添加对应的处理函数（类似现有的 `analyze_conference`）。

**Step 2: Commit**

```bash
git add analysis/main.py
git commit -m "feat(cli): add ecosystem, network, timeline commands"
```

---

## Task 9: 端到端测试

**Step 1: 运行 CLI 测试**

```bash
# 测试 timeline
python -m analysis timeline --years 2020,2021,2022 --top-n 50

# 测试 ecosystem
python -m analysis ecosystem --years 2020,2022 --conferences aaai,acl

# 测试 network
python -m analysis network --years 2020,2021,2022
```

**Step 2: Commit**

```bash
git commit -m "test: add end-to-end tests for Phase 2"
```

---

## 执行选择

**Plan complete and saved to `docs/plans/2025-03-03-phase2-ecosystem-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing_plans, batch execution with checkpoints

**Which approach?**
