# analysis/ngram_extractor.py

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import re

from gensim.models import Phrases
from gensim.models.phrases import Phraser

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
        self._num_phrases: int = 0

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
            # Count phrases from vocabulary
            bigram_phrases = sum(1 for w in self.bigram_model.vocab if '_' in w)
            logger.info(f"Bigram model trained: {bigram_phrases} phrases")

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
            trigram_phrases = sum(1 for w in self.trigram_model.vocab if '_' in w)
            logger.info(f"Trigram model trained: {trigram_phrases} phrases")

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
            # 获取 Bigram 短语 - use export_phrases() in gensim 4.x
            bigram_phrases = self.bigram_model.export_phrases()
            for phrase, score in bigram_phrases.items():
                phrases[phrase.replace('_', ' ')] = float(score)

        if self.trigram_model and self.ngram_type == 'trigram':
            # 获取 Trigram 短语
            trigram_phrases = self.trigram_model.export_phrases()
            for phrase, score in trigram_phrases.items():
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
