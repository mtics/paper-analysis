#!/usr/bin/env python3
"""
Topic modeling and clustering module.
Cluster papers into topics/domains.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Sklearn not available for clustering")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Predefined domain keywords for classification
DOMAIN_KEYWORDS = {
    'Machine Learning': [
        'machine learning', 'deep learning', 'neural network', 'reinforcement learning',
        'supervised learning', 'unsupervised learning', 'semi-supervised', 'transfer learning',
        'few-shot learning', 'zero-shot learning', 'meta-learning', 'representation learning',
        'embedding', 'feature learning', 'gradient descent', 'optimization'
    ],
    'Computer Vision': [
        'computer vision', 'image', 'object detection', 'image segmentation', 'semantic segmentation',
        'instance segmentation', 'object tracking', 'image classification', 'recognition',
        'face recognition', 'pose estimation', '3d reconstruction', 'image generation',
        'gan', 'diffusion', 'image synthesis', 'visual', 'video', 'scene understanding'
    ],
    'Natural Language Processing': [
        'natural language processing', 'nlp', 'language model', 'text', 'word embedding',
        'bert', 'gpt', 'transformer', 'attention', 'seq2seq', 'machine translation',
        'text generation', 'question answering', 'text classification', 'sentiment',
        'named entity recognition', 'ner', 'parsing', 'text summarization', 'dialogue'
    ],
    'Information Retrieval': [
        'information retrieval', 'search', 'recommender', 'recommendation', 'ranking',
        'query', 'retrieval', 'relevance', 'ir system', 'search engine'
    ],
    'Data Mining': [
        'data mining', 'knowledge discovery', 'pattern mining', 'association rule',
        'clustering', 'outlier detection', 'anomaly detection', 'time series',
        'data stream', 'big data', 'data analysis'
    ],
    'Database': [
        'database', 'query', 'sql', 'transaction', 'indexing', 'data management',
        'data storage', 'nosql', 'graph database', 'data integration', 'data quality'
    ],
    'Security & Privacy': [
        'security', 'privacy', 'cryptography', 'encryption', 'authentication',
        'attack', 'defense', 'vulnerability', 'malware', 'blockchain', 'secure'
    ],
    'Software Engineering': [
        'software engineering', 'program', 'code', 'software', 'debugging', 'testing',
        'program analysis', 'program verification', 'compilation', 'programming language'
    ],
    'Human-Computer Interaction': [
        'human-computer interaction', 'hci', 'user interface', 'ui', 'ux',
        'interaction', 'user study', 'usability', 'virtual reality', 'vr', 'ar',
        'augmented reality', 'accessibility'
    ],
    'Systems & Networks': [
        'distributed system', 'cloud computing', 'edge computing', 'networking',
        'protocol', 'server', 'performance', 'optimization', 'parallel', 'distributed',
        'cluster', 'scheduling', 'resource management'
    ],
    'Artificial Intelligence': [
        'artificial intelligence', 'ai', 'agent', 'planning', 'reasoning',
        'knowledge base', 'expert system', 'game', 'robotics', 'autonomous'
    ],
    'Theory & Algorithms': [
        'algorithm', 'complexity', 'theory', 'computational', 'approximation',
        'graph', 'combinatorial', 'optimization', 'mathematical', 'probabilistic'
    ]
}


@dataclass
class Topic:
    """Topic data structure."""
    id: int
    name: str
    keywords: List[str]
    paper_count: int = 0
    papers: List = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'keywords': self.keywords,
            'paper_count': self.paper_count,
            'paper_ids': [p.get('title', '')[:50] for p in self.papers]
        }


class DomainClassifier:
    """Classify papers into predefined domains using keyword matching."""

    def __init__(self, domain_keywords: Dict[str, List[str]] = None):
        """
        Initialize domain classifier.

        Args:
            domain_keywords: Custom domain keywords. If None, use default.
        """
        self.domain_keywords = domain_keywords or DOMAIN_KEYWORDS

        # Create inverted index for faster lookup
        self.keyword_to_domain = {}
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                self.keyword_to_domain[keyword.lower()] = domain

    def classify(self, text: str, min_match: int = 1) -> Optional[str]:
        """
        Classify text into a domain.

        Args:
            text: Text to classify
            min_match: Minimum keyword matches required

        Returns:
            Domain name or None
        """
        if not text:
            return None

        text_lower = text.lower()
        domain_scores = Counter()

        for keyword, domain in self.keyword_to_domain.items():
            if keyword in text_lower:
                domain_scores[domain] += 1

        if not domain_scores or max(domain_scores.values()) < min_match:
            return None

        return domain_scores.most_common(1)[0][0]

    def classify_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]

    def get_all_domains(self) -> List[str]:
        """Get list of all domains."""
        return list(self.domain_keywords.keys())


class TopicModeler:
    """Topic modeling using clustering."""

    def __init__(self, method: str = 'kmeans', n_topics: int = 10, cache_dir: str = None):
        """
        Initialize topic modeler.

        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            n_topics: Number of topics/clusters
            cache_dir: Directory for caching models
        """
        self.method = method
        self.n_topics = n_topics
        self.model = None
        self.vectorizer = None
        self.topic_keywords = {}

        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, texts: List[str]) -> Path:
        """Get cache path for model."""
        text_hash = hashlib.md5(str(texts).encode()).hexdigest()[:16]
        return self.cache_dir / f"topic_model_{self.method}_{self.n_topics}_{text_hash}.pkl"

    def fit_transform(self, texts: List[str], use_cache: bool = True) -> List[int]:
        """
        Fit model and transform texts to topics.

        Args:
            texts: List of preprocessed texts
            use_cache: Whether to use cached model

        Returns:
            List of topic IDs
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Sklearn not available for topic modeling")
            return []

        cache_path = self._get_cache_path(texts)

        # Try to load from cache
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data['vectorizer']
                    self.topic_keywords = data.get('topic_keywords', {})
                    logger.info(f"Loaded topic model from cache: {cache_path}")
                    return data['labels']
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Apply dimensionality reduction
            if tfidf_matrix.shape[1] > 100:
                svd = TruncatedSVD(n_components=100, random_state=42)
                X = svd.fit_transform(tfidf_matrix)
            else:
                X = tfidf_matrix.toarray()

            # Clustering
            if self.method == 'kmeans':
                self.model = KMeans(n_clusters=self.n_topics, random_state=42, n_init=10)
                labels = self.model.fit_predict(X)
            elif self.method == 'dbscan':
                self.model = DBSCAN(eps=0.5, min_samples=5)
                labels = self.model.fit_predict(X)
            else:
                self.model = AgglomerativeClustering(n_clusters=self.n_topics)
                labels = self.model.fit_predict(X)

            # Extract topic keywords
            self._extract_topic_keywords(self.vectorizer, texts, labels)

            # Cache results
            cache_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'labels': labels,
                'topic_keywords': self.topic_keywords
            }

            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached topic model to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")

            return labels

        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return []

    def _extract_topic_keywords(self, vectorizer, texts: List[str], labels: List[int]):
        """Extract top keywords for each topic."""
        feature_names = vectorizer.get_feature_names_out()

        # Get cluster centers
        if hasattr(self.model, 'cluster_centers_'):
            centers = self.model.cluster_centers_

            # For each cluster, get top keywords
            for topic_id in range(len(centers)):
                center = centers[topic_id]
                top_indices = center.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices if center[i] > 0]
                self.topic_keywords[topic_id] = keywords

    def get_topics(self) -> Dict[int, List[str]]:
        """Get all topics with keywords."""
        return self.topic_keywords


class SubtopicAnalyzer:
    """Analyze subtopics within a domain using keyword co-occurrence."""

    def __init__(self, top_n_keywords: int = 50):
        """
        Initialize subtopic analyzer.

        Args:
            top_n_keywords: Number of top keywords to use
        """
        self.top_n_keywords = top_n_keywords

    def analyze(self, papers: List, year_range: Tuple[int, int] = None) -> Dict:
        """
        Analyze subtopics in papers.

        Args:
            papers: List of papers
            year_range: Year range to analyze

        Returns:
            Analysis results with subtopics
        """
        # Filter by year
        if year_range:
            papers = [p for p in papers if year_range[0] <= p.year <= year_range[1]]

        # Extract keywords from all papers
        keyword_counter = Counter()
        keyword_by_year = defaultdict(Counter)

        for paper in papers:
            text = paper.title
            if paper.has_abstract:
                text += " " + paper.abstract

            # Simple keyword extraction
            words = text.lower().split()

            # Filter short words and stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                        'i', 'we', 'they', 'he', 'she', 'it', 'my', 'our', 'their', 'its', 'from',
                        'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                        'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                        'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
                        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'paper', 'approach',
                        'method', 'system', 'result', 'work', 'propose', 'show', 'use', 'new'}

            words = [w for w in words if len(w) > 2 and w not in stopwords]

            # Count keywords
            for word in words:
                keyword_counter[word] += 1
                keyword_by_year[paper.year][word] += 1

        # Get top keywords
        top_keywords = [kw for kw, _ in keyword_counter.most_common(self.top_n_keywords)]

        # Analyze keyword trends over years
        years = sorted(set(p.year for p in papers))
        keyword_trends = {}

        for keyword in top_keywords[:20]:  # Top 20 for trends
            keyword_trends[keyword] = []
            for year in years:
                year_papers = [p for p in papers if p.year == year]
                year_text = " ".join([
                    p.title + (" " + p.abstract if p.has_abstract else "")
                    for p in year_papers
                ])
                count = year_text.lower().count(keyword)
                keyword_trends[keyword].append({'year': year, 'count': count})

        return {
            'top_keywords': top_keywords,
            'keyword_counts': dict(keyword_counter.most_common(50)),
            'keyword_trends': keyword_trends,
            'years': years,
            'total_papers': len(papers)
        }


def classify_papers_by_domain(papers: List) -> Dict[str, List]:
    """
    Classify papers into domains.

    Args:
        papers: List of Paper objects

    Returns:
        Dictionary mapping domain to papers
    """
    classifier = DomainClassifier()
    domain_papers = defaultdict(list)

    for paper in papers:
        text = paper.title
        if paper.has_abstract:
            text += " " + paper.abstract

        domain = classifier.classify(text)
        if domain:
            domain_papers[domain].append(paper)

    return dict(domain_papers)


if __name__ == "__main__":
    # Test domain classification
    classifier = DomainClassifier()

    test_texts = [
        "We propose a novel deep learning method for image classification using convolutional neural networks",
        "This paper presents a new approach to machine translation using transformer architecture",
        "We introduce a new database indexing technique for efficient query processing",
        "Our system detects malicious attacks using machine learning-based anomaly detection"
    ]

    for text in test_texts:
        domain = classifier.classify(text)
        print(f"Text: {text[:60]}...")
        print(f"Domain: {domain}\n")

    print("Available domains:", classifier.get_all_domains())
