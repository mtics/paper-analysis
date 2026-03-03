# tests/test_ecosystem.py

import pytest
from analysis.ecosystem import (
    VocabularyTimeline,
    ConferenceSimilarityMatrix,
    TechnologyDiffusion,
    KnowledgeFlowGraph
)


class TestVocabularyTimeline:

    def test_init_default(self):
        vt = VocabularyTimeline()
        assert vt.min_count == 10
        assert vt.min_total_count == 50

    def test_init_custom(self):
        vt = VocabularyTimeline(min_count=5, min_count_total=100)
        assert vt.min_count == 5
        assert vt.min_total_count == 100


class TestConferenceSimilarityMatrix:

    def test_init_default(self):
        csm = ConferenceSimilarityMatrix()
        assert csm.max_features == 5000


class TestTechnologyDiffusion:

    def test_init_default(self):
        td = TechnologyDiffusion()
        assert td.threshold == 10

    def test_init_custom(self):
        td = TechnologyDiffusion(threshold=5)
        assert td.threshold == 5


class TestKnowledgeFlowGraph:

    def test_init_default(self):
        kfg = KnowledgeFlowGraph()
        assert kfg.pmi_threshold == 0.5

    def test_init_custom(self):
        kfg = KnowledgeFlowGraph(pmi_threshold=0.3)
        assert kfg.pmi_threshold == 0.3
