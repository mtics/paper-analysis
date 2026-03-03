#!/usr/bin/env python3
"""
Text preprocessing module.
Text cleaning, tokenization, and embedding.
"""

import re
import logging
from typing import List, Optional, Dict
from pathlib import Path
import pickle
import hashlib

from analysis.ngram_extractor import NgramExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic preprocessing")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Sklearn not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available")


# Default English stopwords (basic list)
DEFAULT_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    'there', 'then', 'once', 'if', 'because', 'while', 'although', 'though',
    'after', 'before', 'above', 'below', 'between', 'into', 'through',
    'during', 'under', 'again', 'further', 'then', 'show', 'use', 'propose',
    'present', 'paper', 'approach', 'method', 'system', 'result', 'work',
    'based', 'using', 'used', 'new', 'novel', 'different', 'existing',
    'proposed', 'show', 'demonstrate', 'achieve', 'perform', 'state',
    'art', 'given', 'set', 'let', 'consider', 'assume', 'define', 'term'
}


class TextPreprocessor:
    """Text preprocessing for papers."""

    def __init__(self, use_lemmatization: bool = False, min_word_length: int = 2):
        """
        Initialize preprocessor.

        Args:
            use_lemmatization: Use lemmatization (requires NLTK)
            min_word_length: Minimum word length to keep
        """
        self.use_lemmatization = use_lemmatization and NLTK_AVAILABLE
        self.min_word_length = min_word_length
        self.stopwords = DEFAULT_STOPWORDS.copy()

        # Initialize NLTK components
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('corpora/stopwords')
                self.stopwords = set(stopwords.words('english'))
                # Add custom stopwords
                self.stopwords.update(DEFAULT_STOPWORDS)
            except LookupError:
                pass

            if self.use_lemmatization:
                try:
                    nltk.data.find('corpora/wordnet')
                    self.lemmatizer = WordNetLemmatizer()
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                    self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep spaces and alphanumerics
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()
        else:
            tokens = text.split()

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        return [t for t in tokens if t.lower() not in self.stopwords]

    def apply_min_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        return [t for t in tokens if len(t) >= self.min_word_length]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.

        Args:
            tokens: List of tokens

        Returns:
            Lemmatized tokens
        """
        if self.lemmatizer is None:
            return tokens

        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str, remove_stopwords_flag: bool = True) -> str:
        """
        Full preprocessing pipeline.

        Args:
            text: Input text
            remove_stopwords_flag: Whether to remove stopwords

        Returns:
            Preprocessed text
        """
        # Clean
        text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        if remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)

        # Filter by length
        tokens = self.apply_min_length(tokens)

        # Lemmatize
        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)

        return ' '.join(tokens)

    def preprocess_papers(self, papers: List[Dict], text_field: str = 'text_content') -> List[str]:
        """
        Preprocess a list of papers.

        Args:
            papers: List of paper dictionaries
            text_field: Field to use for text

        Returns:
            List of preprocessed texts
        """
        texts = []

        for paper in papers:
            if text_field == 'text_content':
                # Get combined title and abstract
                text = getattr(paper, 'text_content', paper.get('title', ''))
            else:
                text = paper.get(text_field, '')

            if text:
                processed = self.preprocess(text)
                texts.append(processed)

        return texts


class EmbeddingGenerator:
    """Generate embeddings for papers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of sentence transformer model
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.model = None

        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        else:
            logger.warning("SentenceTransformers not available, using TF-IDF fallback")

    def _get_cache_path(self, texts: List[str]) -> Path:
        """Get cache file path for texts."""
        # Create hash of texts
        text_hash = hashlib.md5(str(texts).encode()).hexdigest()[:16]
        return self.cache_dir / f"embeddings_{self.model_name}_{text_hash}.pkl"

    def encode(self, texts: List[str], use_cache: bool = True) -> Optional[List[List[float]]]:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            use_cache: Whether to use cached embeddings

        Returns:
            List of embeddings or None
        """
        if not texts:
            return []

        cache_path = self._get_cache_path(texts)

        # Try to load from cache
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    if len(cached) == len(texts):
                        logger.info(f"Loaded embeddings from cache: {cache_path}")
                        return cached
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Generate embeddings
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, show_progress_bar=True)

                # Cache results
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embeddings, f)
                    logger.info(f"Cached embeddings to: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache embeddings: {e}")

                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")

        # Fallback to TF-IDF
        return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: List[str], max_features: int = 512) -> Optional[List[List[float]]]:
        """
        Fallback TF-IDF encoding.

        Args:
            texts: List of texts
            max_features: Maximum number of features

        Returns:
            TF-IDF vectors
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Sklearn not available for TF-IDF fallback")
            return None

        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(texts)

            # Reduce dimensionality
            svd = TruncatedSVD(n_components=min(256, tfidf_matrix.shape[1] - 1))
            reduced = svd.fit_transform(tfidf_matrix)

            return reduced.tolist()
        except Exception as e:
            logger.error(f"TF-IDF encoding failed: {e}")
            return None


class KeywordExtractor:
    """Extract keywords from papers."""

    def __init__(self, top_n: int = 10):
        """
        Initialize keyword extractor.

        Args:
            top_n: Number of keywords to extract
        """
        self.top_n = top_n
        self.vectorizer = None

    def extract(self, texts: List[str], use_tfidf: bool = True) -> List[List[tuple]]:
        """
        Extract keywords from texts.

        Args:
            texts: List of texts
            use_tfidf: Use TF-IDF for scoring

        Returns:
            List of keyword lists (keyword, score)
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Sklearn not available for keyword extraction")
            return [[] for _ in texts]

        try:
            # Use TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )

            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            keywords = []

            for i in range(tfidf_matrix.shape[0]):
                row = tfidf_matrix[i].toarray().flatten()
                top_indices = row.argsort()[-self.top_n:][::-1]

                top_keywords = [
                    (feature_names[idx], float(row[idx]))
                    for idx in top_indices if row[idx] > 0
                ]

                keywords.append(top_keywords)

            return keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return [[] for _ in texts]


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


def preprocess_papers(papers: List, add_abstract: bool = True) -> List[str]:
    """
    Convenience function to preprocess papers.

    Args:
        papers: List of Paper objects
        add_abstract: Whether to include abstract

    Returns:
        List of preprocessed texts
    """
    preprocessor = TextPreprocessor()
    texts = []

    for paper in papers:
        # Combine title and abstract
        text = paper.title
        if add_abstract and paper.has_abstract:
            text += " " + paper.abstract

        processed = preprocessor.preprocess(text)
        texts.append(processed)

    return texts


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()

    test_text = "This paper proposes a novel deep learning approach for natural language processing tasks. We demonstrate improved performance on several benchmark datasets."

    processed = preprocessor.preprocess(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")

    # Test keyword extraction
    texts = [
        "deep learning neural network transformer attention mechanism",
        "machine learning classification regression supervised unsupervised",
        "natural language processing BERT GPT language model"
    ]

    extractor = KeywordExtractor(top_n=5)
    keywords = extractor.extract(texts)

    for i, kw in enumerate(keywords):
        print(f"\nText {i+1} keywords: {kw}")
