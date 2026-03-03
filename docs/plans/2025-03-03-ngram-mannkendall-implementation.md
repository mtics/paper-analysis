# 阶段一：N-gram + Mann-Kendall 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 N-gram 短语提取（支持 trigram）和 Mann-Kendall 趋势检验，作为独立模块并集成到现有代码

**Architecture:**
- 新建 `ngram_extractor.py`：基于 Gensim Phrases 的 N-gram 提取器
- 新建 `stats_utils.py`：Mann-Kendall 趋势检验工具
- 修改 `preprocessing.py`：集成 N-gram 预处理
- 修改 `trend_analysis.py`：使用 Mann-Kendall 替代朴素增长率

**Tech Stack:** gensim, pymannkendall, nltk, pandas

---

### Task 1: 创建 stats_utils.py - Mann-Kendall 趋势检验

**Files:**
- Create: `analysis/stats_utils.py`

**Step 1: 创建基础框架**

```python
# analysis/stats_utils.py

import pymannkendall as mk
from typing import Dict, List, Tuple, Optional


def mann_kendall_test(yearly_counts: Dict[int, int]) -> Dict:
    """
    对年度论文数量进行 Mann-Kendall 趋势检验

    Args:
        yearly_counts: {年份: 论文数量}

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
        'significant': result.p < 0.05,
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


def calculate_growth_rate(yearly_counts: Dict[int, int], method: str = 'compound') -> float:
    """
    计算复合年增长率 (CAGR)

    Args:
        yearly_counts: {年份: 论文数量}
        method: 'compound' | 'simple'

    Returns:
        年均增长率 (如 0.15 表示 15%)
    """
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
```

**Step 2: 验证实现**

```bash
# 测试代码
python -c "
from analysis.stats_utils import mann_kendall_test

# 测试数据
yearly = {2017: 12, 2018: 45, 2019: 120, 2020: 280, 2021: 450}
result = mann_kendall_test(yearly)
print(result)
# Expected: trend='increasing', significant=True, p_value < 0.05
"
```

**Step 3: Commit**

```bash
git add analysis/stats_utils.py
git commit -m "feat: add Mann-Kendall trend test utilities"
```

---

### Task 2: 创建 ngram_extractor.py - N-gram 短语提取器

**Files:**
- Create: `analysis/ngram_extractor.py`

**Step 1: 创建基础框架**

```python
# analysis/ngram_extractor.py

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import re

import nltk
from gensim.models import Phrases, Phraser
from gensim.models.phrases import Normalizer

logger = logging.getLogger(__name__)


class NgramExtractor:
    """使用 Gensim Phrases 提取 N-gram 短语"""

    def __init__(
        self,
        min_count: int = 5,
        threshold: float = 10.0,
        ngram_type: str = 'trigram'
    ):
        """
        Args:
            min_count: 最小词频阈值，低于此值的词组会被忽略
            threshold: 评分阈值，较高的值产生更严格的短语提取
            ngram_type: 'bigram' | 'trigram'
        """
        self.min_count = min_count
        self.threshold = threshold
        self.ngram_type = ngram_type

        self.bigram_model: Optional[Phrases] = None
        self.trigram_model: Optional[Phrases] = None
        self.bigram_phraser: Optional[Phraser] = None
        self.trigram_phraser: Optional[Phraser] = None

        self._vocabulary: Set[str] = set()

    def _tokenize(self, text: str) -> List[str]:
        """基础分词 - 转为小写并分词"""
        if not text:
            return []
        # 简单分词：转小写，移除标点，空格分词
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)
        # 过滤过短的词
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    def fit(self, texts: List[str]) -> 'NgramExtractor':
        """
        训练 N-gram 模型

        Args:
            texts: 原始文本列表（每篇论文的 title + abstract）
        """
        logger.info(f"Training {self.ngram_type} extractor on {len(texts)} texts...")

        # 第一步：分词
        tokenized = [self._tokenize(t) for t in texts]
        # 过滤空文本
        tokenized = [t for t in tokenized if len(t) > 0]

        if self.ngram_type in ['bigram', 'trigram']:
            # 训练 Bigram 模型
            self.bigram_model = Phrases(
                tokenized,
                min_count=self.min_count,
                threshold=self.threshold,
                scoring='npmi'  # 使用 NPMI 评分，更稳定
            )
            self.bigram_phraser = Phraser(self.bigram_model)
            logger.info(f"Bigram model trained: {len(self.bigram_model.phrase_docs)} phrases")

        if self.ngram_type == 'trigram':
            # 第二步：用 Bigram 结果训练 Trigram
            bigram_transformed = [self.bigram_model[sent] for sent in tokenized]
            self.trigram_model = Phrases(
                bigram_transformed,
                min_count=self.min_count,
                threshold=self.threshold,
                scoring='npmi'
            )
            self.trigram_phraser = Phraser(self.trigram_model)
            logger.info(f"Trigram model trained: {len(self.trigram_model.phrase_docs)} phrases")

        # 收集词汇表
        for tokens in tokenized:
            self._vocabulary.update(tokens)

        return self

    def transform(self, texts: List[str]) -> List[List[str]]:
        """
        将文本转换为包含 N-gram 的词列表

        Args:
            texts: 原始文本列表

        Returns:
            每个文本对应的词列表，包含检测到的短语
        """
        if not texts:
            return []

        tokenized = [self._tokenize(t) for t in texts]

        if self.ngram_type == 'bigram':
            return [list(self.bigram_phraser[sent]) for sent in tokenized]
        elif self.ngram_type == 'trigram':
            # 先应用 bigram，再用 trigram
            bigrammed = [self.bigram_phraser[sent] for sent in tokenized]
            return [list(self.trigram_phraser[sent]) for sent in bigrammed]
        else:
            return tokenized

    def get_phrases(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        获取提取的短语及其评分

        Args:
            top_n: 返回前 N 个短语，None 表示全部

        Returns:
            {短语: 评分}
        """
        phrases = {}

        if self.bigram_model and self.ngram_type in ['bigram', 'trigram']:
            # 获取 Bigram 短语
            for phrase, score in self.bigram_model.find_phrases(self._vocabulary).items():
                if '_' in phrase:  # Gensim 用下划线连接短语
                    phrases[phrase.replace('_', ' ')] = float(score)

        if self.trigram_model and self.ngram_type == 'trigram':
            # 获取 Trigram 短语
            for phrase, score in self.trigram_model.find_phrases(self._vocabulary).items():
                if '_' in phrase:
                    phrases[phrase.replace('_', ' ')] = float(score)

        # 按评分排序
        sorted_phrases = dict(sorted(phrases.items(), key=lambda x: x[1], reverse=True))

        if top_n:
            return dict(list(sorted_phrases.items())[:top_n])
        return sorted_phrases

    def extract_keyphrases(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        从单篇论文中提取关键短语

        Args:
            text: 论文文本
            top_k: 返回前 K 个短语

        Returns:
            [(短语, 出现次数), ...]
        """
        tokens = self.transform([text])[0]
        # 统计短语出现次数
        phrase_counter = Counter()
        for token in tokens:
            if '_' in token:
                phrase_counter[token.replace('_', ' ')] += 1

        return phrase_counter.most_common(top_k)


# 便捷函数
def extract_ngrams(
    texts: List[str],
    ngram_type: str = 'trigram',
    min_count: int = 5,
    threshold: float = 10.0
) -> Tuple[NgramExtractor, List[List[str]]]:
    """
    便捷函数：一步完成训练和转换

    Args:
        texts: 文本列表
        ngram_type: 'bigram' | 'trigram'
        min_count: 最小词频
        threshold: 评分阈值

    Returns:
        (训练好的模型, 转换后的词列表)
    """
    extractor = NgramExtractor(
        min_count=min_count,
        threshold=threshold,
        ngram_type=ngram_type
    )
    extractor.fit(texts)
    transformed = extractor.transform(texts)
    return extractor, transformed
```

**Step 2: 验证实现**

```bash
# 测试代码
python -c "
from analysis.ngram_extractor import NgramExtractor

# 测试数据
texts = [
    'large language model is powerful',
    'transformer based language models',
    'neural network architecture',
    'large language models for reasoning',
    'transformer architecture in NLP'
]

extractor = NgramExtractor(min_count=1, threshold=1.0, ngram_type='bigram')
extractor.fit(texts)
phrases = extractor.get_phrases()
print('Phrases:', phrases)

transformed = extractor.transform(texts)
print('Transformed:', transformed)
"
```

**Step 3: Commit**

```bash
git add analysis/ngram_extractor.py
git commit -m "feat: add N-gram phrase extractor (bigram/trigram)"
```

---

### Task 3: 创建测试文件

**Files:**
- Create: `tests/test_ngram_extractor.py`
- Create: `tests/test_stats_utils.py`

**Step 1: 创建测试文件**

```python
# tests/test_ngram_extractor.py

import pytest
from analysis.ngram_extractor import NgramExtractor, extract_ngrams


class TestNgramExtractor:

    def test_bigram_extraction(self):
        texts = [
            'large language model',
            'machine learning',
            'deep learning',
            'neural network'
        ]
        extractor = NgramExtractor(min_count=1, threshold=1.0, ngram_type='bigram')
        extractor.fit(texts)
        phrases = extractor.get_phrases()
        assert isinstance(phrases, dict)

    def test_trigram_extraction(self):
        texts = [
            'large language model',
            'reinforcement learning from human feedback',
            'natural language processing'
        ]
        extractor = NgramExtractor(min_count=1, threshold=1.0, ngram_type='trigram')
        extractor.fit(texts)
        phrases = extractor.get_phrases()
        assert 'large language model' in phrases or 'reinforcement learning from' in phrases

    def test_transform_returns_list(self):
        texts = ['hello world', 'test text']
        extractor = NgramExtractor(ngram_type='bigram')
        extractor.fit(texts)
        result = extractor.transform(texts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_keyphrases(self):
        text = 'large language model is powerful large language model'
        extractor = NgramExtractor(min_count=1, threshold=1.0, ngram_type='bigram')
        extractor.fit([text])
        keyphrases = extractor.extract_keyphrases(text, top_k=5)
        assert len(keyphrases) <= 5


class TestConvenienceFunction:

    def test_extract_ngrams(self):
        texts = ['test text', 'sample content']
        extractor, transformed = extract_ngrams(texts, ngram_type='bigram')
        assert isinstance(extractor, NgramExtractor)
        assert isinstance(transformed, list)
```

```python
# tests/test_stats_utils.py

import pytest
from analysis.stats_utils import mann_kendall_test, normalize_yearly_counts, calculate_growth_rate


class TestMannKendall:

    def test_increasing_trend(self):
        yearly = {2017: 10, 2018: 30, 2019: 80, 2020: 200, 2021: 400}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'increasing'
        assert result['significant'] == True
        assert result['p_value'] < 0.05

    def test_decreasing_trend(self):
        yearly = {2017: 400, 2018: 300, 2019: 200, 2020: 100, 2021: 50}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'decreasing'
        assert result['significant'] == True

    def test_insufficient_data(self):
        yearly = {2020: 100, 2021: 110}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'no trend'
        assert result['significant'] == False

    def test_empty_data(self):
        yearly = {}
        result = mann_kendall_test(yearly)
        assert result['trend'] == 'no trend'


class TestNormalizeYearly:

    def test_normalize(self):
        yearly = {2020: 100, 2021: 200, 2022: 300}
        normalized = normalize_yearly_counts(yearly)
        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert normalized[2020] == 0.1666


class TestGrowthRate:

    def test_compound_growth(self):
        yearly = {2020: 100, 2021: 150, 2022: 225}  # 50% growth each year
        rate = calculate_growth_rate(yearly, method='compound')
        assert abs(rate - 0.50) < 0.01

    def test_zero_growth(self):
        yearly = {2020: 100, 2021: 100, 2022: 100}
        rate = calculate_growth_rate(yearly)
        assert abs(rate) < 0.01
```

**Step 2: 运行测试**

```bash
pytest tests/test_ngram_extractor.py tests/test_stats_utils.py -v
```

**Step 3: Commit**

```bash
git add tests/test_ngram_extractor.py tests/test_stats_utils.py
git commit -m "test: add unit tests for N-gram and stats utils"
```

---

### Task 4: 集成到 preprocessing.py

**Files:**
- Modify: `analysis/preprocessing.py`

**Step 1: 添加 NgramPreprocessor 类**

在文件末尾添加：

```python
class NgramPreprocessor:
    """带 N-gram 支持的文本预处理器"""

    def __init__(self, ngram_type: str = 'trigram', min_count: int = 5, threshold: float = 10.0):
        self.ngram_type = ngram_type
        self.min_count = min_count
        self.threshold = threshold
        self.extractor: Optional[NgramExtractor] = None

    def fit(self, texts: List[str]) -> 'NgramPreprocessor':
        """训练 N-gram 模型"""
        self.extractor = NgramExtractor(
            ngram_type=self.ngram_type,
            min_count=self.min_count,
            threshold=self.threshold
        )
        self.extractor.fit(texts)
        return self

    def transform(self, texts: List[str]) -> List[List[str]]:
        """转换文本为包含 N-gram 的词列表"""
        if self.extractor is None:
            raise ValueError("Must call fit() before transform()")
        return self.extractor.transform(texts)

    def fit_transform(self, texts: List[str]) -> List[List[str]]:
        """一步完成训练和转换"""
        self.fit(texts)
        return self.transform(texts)

    def get_phrases(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """获取提取的短语"""
        if self.extractor is None:
            raise ValueError("Must call fit() before get_phrases()")
        return self.extractor.get_phrases(top_n)
```

**Step 2: 更新 imports**

确保文件顶部有：
```python
from analysis.ngram_extractor import NgramExtractor
```

**Step 3: 验证**

```bash
python -c "
from analysis.preprocessing import NgramPreprocessor

texts = ['large language model', 'neural network architecture']
prep = NgramPreprocessor(ngram_type='bigram', min_count=1, threshold=1.0)
result = prep.fit_transform(texts)
print(result)
phrases = prep.get_phrases()
print('Phrases:', phrases)
"
```

**Step 4: Commit**

```bash
git add analysis/preprocessing.py
git commit -m "feat: integrate N-gram extractor into preprocessing"
```

---

### Task 5: 集成到 trend_analysis.py

**Files:**
- Modify: `analysis/trend_analysis.py`

**Step 1: 找到并修改 _analyze_emerging_topics 方法**

在 `_analyze_emerging_topics` 方法中，找到朴素的增长率计算：

```python
# 原代码（需要替换）
growth_rates = []
for word, yearly in keyword_by_year.items():
    if len(yearly) >= 2:
        early = sum(yearly.get(y, 0) for y in early_years)
        late = sum(yearly.get(y, 0) for y in late_years)
        if early > 0:
            growth = (late - early) / early
            growth_rates.append((word, growth))
```

替换为：

```python
# 新代码（使用 Mann-Kendall）
from analysis.stats_utils import mann_kendall_test

growth_rates = []
for word, yearly in keyword_by_year.items():
    if len(yearly) >= 3:  # Mann-Kendall 至少需要 3 个数据点
        # 转换为 {year: count} 格式
        yearly_counts = {int(y): c for y, c in yearly.items()}

        # Mann-Kendall 趋势检验
        mk_result = mann_kendall_test(yearly_counts)

        if mk_result['significant'] and mk_result['trend'] == 'increasing':
            growth_rates.append({
                'word': word,
                'growth_rate': mk_result['sens_slope'],  # 使用 Sen's 斜率
                'p_value': mk_result['p_value'],
                'trend': mk_result['trend']
            })

# 按增长率排序
growth_rates.sort(key=lambda x: x['growth_rate'], reverse=True)
```

**Step 2: 确保导入**

在文件顶部添加：
```python
from analysis.stats_utils import mann_kendall_test
```

**Step 3: 验证**

```bash
python -m analysis analyze aaai --years 2020,2021,2022 --emerging
```

**Step 4: Commit**

```bash
git add analysis/trend_analysis.py
git commit -m "feat: integrate Mann-Kendall trend test in emerging topics"
```

---

### Task 6: 端到端集成测试

**Step 1: 创建 CLI 命令测试**

```bash
# 测试完整流程
python -m analysis analyze aaai --years 2018,2019,2020,2021,2022 --keywords --emerging
```

预期输出应包含：
- 关键词统计
- 显著新兴趋势（带 p_value）

**Step 2: Commit**

```bash
git commit -m "test: add end-to-end integration test for N-gram + Mann-Kendall"
```

---

## 执行选择

**Plan complete and saved to `docs/plans/2025-03-03-ngram-mannkendall-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
