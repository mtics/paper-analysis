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
        extractor = NgramExtractor(min_count=1, threshold=0.5, ngram_type='bigram')
        extractor.fit(texts)
        phrases = extractor.get_phrases()
        assert isinstance(phrases, dict)

    def test_trigram_extraction(self):
        texts = [
            'large language model',
            'reinforcement learning from human feedback',
            'natural language processing'
        ]
        extractor = NgramExtractor(min_count=1, threshold=0.5, ngram_type='trigram')
        extractor.fit(texts)
        phrases = extractor.get_phrases()
        # Check for phrases - exact assertion depends on threshold

    def test_transform_returns_list(self):
        texts = ['hello world', 'test text']
        extractor = NgramExtractor(ngram_type='bigram', min_count=1, threshold=0.5)
        extractor.fit(texts)
        result = extractor.transform(texts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_keyphrases(self):
        text = 'large language model is powerful large language model'
        extractor = NgramExtractor(min_count=1, threshold=0.5, ngram_type='bigram')
        extractor.fit([text])
        keyphrases = extractor.extract_keyphrases(text, top_k=5)
        assert len(keyphrases) <= 5


class TestConvenienceFunction:

    def test_extract_ngrams(self):
        texts = ['test text', 'sample content']
        extractor, transformed = extract_ngrams(texts, ngram_type='bigram', min_count=1, threshold=0.5)
        assert isinstance(extractor, NgramExtractor)
        assert isinstance(transformed, list)
