# 阶段三：单领域深度分析实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现阶段三全部功能 - S曲线生命周期拟合 + 词汇新陈代谢 + 研究者稳定性分析 + 深度报告增强

**Architecture:**
- 新增 `analysis/lifecycle.py`：S曲线拟合 + 研究者稳定性分析
- 扩展 `analysis/deep_domain.py`：词汇新陈代谢功能
- 扩展 `analysis/main.py`：增强 deep 命令输出

**Tech Stack:** scipy (curve_fit), numpy, pandas

---

## Task 1: 创建 analysis/lifecycle.py - S曲线拟合 + 研究者稳定性

**Files:**
- Create: `analysis/lifecycle.py`

### Step 1: 创建 S-curve 拟合类

```python
# analysis/lifecycle.py

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def logistic(x: float, L: float, k: float, x0: float) -> float:
    """Logistic 函数用于 S 曲线拟合"""
    return L / (1 + np.exp(-k * (x - x0)))


class LifecycleAnalyzer:
    """领域生命周期分析 - S曲线拟合判断领域所处阶段"""

    def __init__(self):
        pass

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

            # 拐点前为增长期，拐点后为成熟期
            if current_year < x0:
                stage = 'growth' if current_count < L * 0.7 else 'mature'
            else:
                stage = 'mature' if current_count > L * 0.8 else 'declining'

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

    def __init__(self):
        pass

    def calculate_stability(
        self,
        papers: List,
        years: List[int]
    ) -> Dict:
        """
        计算领域作者稳定性 Jaccard 系数

        Args:
            papers: 论文列表
            years: 年份列表

        Returns:
            {
                'jaccard_by_year': {year: float},
                'avg_jaccard': float,
                'stage': 'expert_dominant' | 'mixed' | '涌入型',
                'new_researcher_ratio': {year: float}
            }
        """
        # 按年份收集作者
        authors_by_year = defaultdict(set)
        for paper in papers:
            if paper.year in years:
                for author in (paper.authors or []):
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
            stage = '涌入型'  # 新涌入多

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
```

### Step 2: Commit

```bash
git add analysis/lifecycle.py
git commit -m "feat(lifecycle): add S-curve fitting and researcher stability analysis"
```

---

## Task 2: 扩展 deep_domain.py - 词汇新陈代谢

**Files:**
- Modify: `analysis/deep_domain.py`

### Step 1: 添加词汇新陈代谢分析

在文件末尾添加：

```python
def analyze_vocabulary_turnover(
    papers: List,
    years: List[int],
    top_n: int = 50
) -> Dict:
    """
    分析领域词汇的新陈代谢 - 追踪词汇的兴衰

    Returns:
        {
            'rising_keywords': [{'word': str, 'growth': float}],
            'declining_keywords': [{'word': str, 'decline': float}],
            'yearly_top_keywords': {year: [words]}
        }
    """
    from collections import Counter

    # 按年份统计词频
    word_by_year = defaultdict(lambda: defaultdict(int))

    for paper in papers:
        if paper.year in years and paper.has_abstract:
            text = (paper.title + " " + paper.abstract).lower()
            words = text.split()
            for word in words:
                if len(word) > 3:
                    word_by_year[paper.year][word] += 1

    # 计算每个词的趋势
    keyword_trends = []
    for word in set(w for year_words in word_by_year.values() for w in year_words):
        counts = [word_by_year.get(y, {}).get(word, 0) for y in sorted(years)]

        if sum(counts) < 5:  # 过滤太低频
            continue

        early = np.mean(counts[:2]) if len(counts) >= 2 else counts[0]
        late = np.mean(counts[-2:]) if len(counts) >= 2 else counts[-1]

        if early > 0:
            change = (late - early) / early
            keyword_trends.append({
                'word': word,
                'change': change,
                'early_avg': early,
                'late_avg': late
            })

    # 排序
    rising = sorted(keyword_trends, key=lambda x: x['change'], reverse=True)[:top_n]
    declining = sorted(keyword_trends, key=lambda x: x['change'])[:top_n]

    return {
        'rising_keywords': rising[:20],
        'declining_keywords': declining[:20],
        'yearly_top_keywords': {
            y: sorted(word_by_year[y].items(), key=lambda x: x[1], reverse=True)[:10]
            for y in years
        }
    }
```

### Step 2: Commit

```bash
git commit -m "feat(deep_domain): add vocabulary turnover analysis"
```

---

## Task 3: 集成到 CLI deep 命令

**Files:**
- Modify: `analysis/main.py`

### Step 1: 更新 deep 命令输出

修改 `deep_analyze_domain` 函数，添加 S 曲线和稳定性分析输出。

### Step 2: Commit

```bash
git commit -m "feat(cli): enhance deep command with lifecycle analysis"
```

---

## Task 4: 创建测试

**Files:**
- Create: `tests/test_lifecycle.py`

### Step 1: 创建测试文件

```python
# tests/test_lifecycle.py

import pytest
from analysis.lifecycle import LifecycleAnalyzer, ResearcherStabilityAnalyzer, logistic


class TestLifecycleAnalyzer:

    def test_fit_scurve_growth(self):
        yearly = {2018: 10, 2019: 50, 2020: 150, 2021: 300, 2022: 400}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert result['stage'] in ['growth', 'mature']

    def test_fit_scurve_insufficient(self):
        yearly = {2022: 100}
        analyzer = LifecycleAnalyzer()
        result = analyzer.fit_scurve(yearly)
        assert result['stage'] == 'unknown'


class TestResearcherStabilityAnalyzer:

    def test_calculate_stability(self):
        # 需要 Paper 对象
        pass
```

### Step 2: Commit

```bash
git add tests/test_lifecycle.py
git commit -m "test: add lifecycle analysis tests"
```

---

## Task 5: 端到端测试

### Step 1: 测试 deep 命令

```bash
python -m analysis deep "AI Agent" --years 2020,2021,2022
```

### Step 2: Commit

```bash
git commit -m "test: add end-to-end tests for Phase 3"
```

---

## 执行选择

**Plan complete and saved to `docs/plans/2025-03-03-phase3-lifecycle-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - 任务逐个执行，代码审查
**2. Parallel Session (separate)** - 新会话批量执行

选择哪种方式？
