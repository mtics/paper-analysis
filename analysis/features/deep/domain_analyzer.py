#!/usr/bin/env python3
"""
Deep domain analysis module.
Provides in-depth analysis for specific research domains.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Domain-specific keywords and sub-topic definitions
DOMAIN_DEFINITIONS = {
    'AI Agent': {
        'keywords': [
            'agent', 'multi-agent', 'multiagent', 'agentic', 'agent system',
            'autonomous agent', 'agent model', 'agent architecture', 'agent framework',
            'gpt-agent', 'llm-agent', 'agentic workflow', 'agent planning',
            'tool use', 'tool calling', 'function calling', 'agent reasoning',
            'reAct', 'reflexion', 'agent memory', 'agent collaboration'
        ],
        'subdomains': {
            'Agent Architecture': [
                'agent architecture', 'agent framework', 'agent design', 'system agent',
                'cognitive architecture', 'modular agent', 'agent system'
            ],
            'Multi-Agent Systems': [
                'multi-agent', 'multiagent', 'multi-agent cooperation', 'multi-agent collaboration',
                'agent communication', 'agent negotiation', 'agent coordination', 'swarm intelligence'
            ],
            'Agent Reasoning': [
                'agent reasoning', 'agent planning', 'chain of thought', 'tool planning',
                'task decomposition', 'self-reflection', 'reAct', 'reflexion', 'ReWOO'
            ],
            'Agent Tool Use': [
                'tool use', 'tool calling', 'function calling', 'tool integration',
                'tool learning', 'tool augmented', 'rag', 'retrieval augmented'
            ],
            'Agent Applications': [
                'agent application', 'agent for code', 'agent for science', 'agent for math',
                'agent for web', 'agent for database', 'software engineering agent'
            ],
            'Agent Evaluation': [
                'agent benchmark', 'agent evaluation', 'agent performance', 'agent capability',
                'agent safety', 'agent alignment', 'agent reliability'
            ]
        }
    },
    'Federated Learning': {
        'keywords': [
            'federated learning', 'federated', 'fl', 'federated training',
            'distributed learning', 'privacy-preserving', 'federated optimization',
            'federated averaging', 'fedavg', 'federated analytics'
        ],
        'subdomains': {
            'Federated Optimization': [
                'federated optimization', 'fedavg', 'fedprox', 'scaffold', 'fednova',
                'personalized fl', 'meta-learning fl', 'continual fl'
            ],
            'Privacy & Security': [
                'privacy', 'differential privacy', 'secure aggregation', 'byzantine',
                'adversarial', 'attack', 'defense', 'homomorphic encryption'
            ],
            'Communication Efficiency': [
                'communication efficient', 'compression', 'quantization', 'sparsification',
                'local training', 'client selection'
            ],
            'Non-IID Data': [
                'non-iid', 'heterogeneous', 'label skew', 'feature skew', 'domain adaptation'
            ],
            'Vertical Federated Learning': [
                'vertical', 'feature-based', 'heterogeneous features', 'cross-silo'
            ],
            'Applications': [
                'federated healthcare', 'federated mobile', 'federated IoT',
                'federated recommendation', 'federated vision'
            ]
        }
    },
    'Large Language Models': {
        'keywords': [
            'llm', 'large language model', 'gpt', 'gpt-4', 'gpt-3', 'language model',
            'transformer language', 'pre-trained model', 'foundation model'
        ],
        'subdomains': {
            'LLM Training': [
                'pretraining', 'pre-training', 'fine-tuning', 'instruction tuning',
                'rlhf', 'dpo', 'alignment', 'continual pretraining'
            ],
            'LLM Reasoning': [
                'reasoning', 'chain of thought', 'cot', 'tree of thought', 'tot',
                'problem solving', 'mathematical reasoning', 'logical reasoning'
            ],
            'LLM Knowledge': [
                'knowledge', 'knowledge editing', 'knowledge memorization',
                'knowledge retrieval', 'parametric knowledge', 'factuality'
            ],
            'LLM Evaluation': [
                'benchmark', 'evaluation', 'capability', 'helpline', 'mt-bench',
                'human evaluation', 'automatic evaluation'
            ],
            'LLM Applications': [
                'application', 'question answering', 'text generation', 'summarization',
                'translation', 'dialogue', 'chatbot'
            ],
            'LLM Efficiency': [
                'efficient', 'quantization', 'distillation', 'pruning', 'acceleration',
                'inference', 'memory efficient', 'low-rank'
            ]
        }
    },
    'Retrieval-Augmented Generation': {
        'keywords': [
            'rag', 'retrieval augmented', 'retrieval-augmented', 'rag pipeline',
            'retrieval generation', 'knowledge retrieval', 'document retrieval',
            'dense retrieval', 'sparse retrieval', 'hybrid retrieval'
        ],
        'subdomains': {
            'Retrieval': [
                'retriever', 'retrieval', 'dense retrieval', 'sparse retrieval',
                'bm25', 'colbert', 'bi-encoder', 'cross-encoder', 'rerank'
            ],
            'Generation': [
                'generator', 'generation', 'llm generation', 'response generation'
            ],
            'RAG Optimization': [
                'rag optimization', 'rag pipeline', 'rag benchmark', 'rag evaluation',
                'chunking', 'embedding', 'indexing'
            ],
            'Multimodal RAG': [
                'multimodal rag', 'visual rag', 'image rag', 'video rag', 'audio rag'
            ],
            'Applications': [
                'rag application', 'rag for qa', 'rag for knowledge', 'rag for code',
                'rag for science', 'domain-specific rag'
            ]
        }
    },
    'Multimodal Learning': {
        'keywords': [
            'multimodal', 'vision language', 'vlm', 'visual language', 'image-text',
            'video understanding', 'audio visual', 'cross-modal', 'multimodal model'
        ],
        'subdomains': {
            'Vision-Language Models': [
                'vlm', 'vision language', 'clip', 'blip', 'llava', 'visual reasoning',
                'image caption', 'visual question answering', 'vqa'
            ],
            'Multimodal Generation': [
                'image generation', 'text-to-image', 'diffusion', 'stable diffusion',
                'text-to-video', 'video generation', 'image-to-image'
            ],
            'Multimodal Understanding': [
                'multimodal understanding', 'multimodal reasoning', 'multimodal learning',
                'audio-visual', 'video understanding', 'action recognition'
            ],
            'Multimodal Alignment': [
                'multimodal alignment', 'cross-modal', 'modal alignment', 'contrastive'
            ]
        }
    },
    'Neural Network Architecture': {
        'keywords': [
            'neural network', 'architecture', 'transformer', 'attention', 'cnn',
            'resnet', 'efficient', 'network design', 'model design'
        ],
        'subdomains': {
            'Transformer Variants': [
                'transformer', 'attention', 'self-attention', 'multi-head attention',
                'vit', 'bert', 'gpt', 'encoder', 'decoder'
            ],
            'Efficient Networks': [
                'efficient', 'lightweight', 'mobile', 'pruning', 'distillation',
                'quantization', 'efficient transformer', 'linear attention'
            ],
            'Network Architecture': [
                'network architecture', 'model architecture', 'resnet', 'unet',
                'gan', 'encoder-decoder', 'graph neural network', 'gnn'
            ]
        }
    },
    'Graph Neural Networks': {
        'keywords': [
            'graph neural network', 'gnn', 'graph convolution', 'graph attention',
            'graph embedding', 'graph learning', 'message passing', 'graph classification'
        ],
        'subdomains': {
            'GNN Architecture': [
                'gcn', 'gat', 'graph convolutional', 'graph attention', 'message passing',
                'graph pooling', 'graph isomorphism'
            ],
            'GNN Applications': [
                'graph matching', 'link prediction', 'node classification', 'graph generation',
                'knowledge graph', 'molecular graph', 'social network'
            ],
            'GNN Theory': [
                'expressivity', 'over-smoothing', 'oversmoothing', 'theoretical analysis',
                'generalization', 'graph structure'
            ]
        }
    },
    'Recommendation Systems': {
        'keywords': [
            'recommendation', 'recommender', 'collaborative filtering', 'content-based',
            'recommender system', 'recsys', 'personalized recommendation'
        ],
        'subdomains': {
            'CF Methods': [
                'collaborative filtering', 'matrix factorization', 'neural cf',
                'graph recommendation', 'session-based'
            ],
            'Sequential Recommendation': [
                'sequential recommendation', 'next-item', 'user behavior', 'sequential modeling'
            ],
            'Graph-based Rec': [
                'graph recommendation', 'knowledge graph recommendation', 'heterogeneous graph'
            ],
            'LLM for Rec': [
                'llm recommendation', 'language model recommendation', 'generative recommendation'
            ]
        }
    },
    'Diffusion Models': {
        'keywords': [
            'diffusion', 'diffusion model', 'ddpm', 'ddim', 'stable diffusion',
            'image generation', 'text-to-image', 'generative model'
        ],
        'subdomains': {
            'Image Generation': [
                'text-to-image', 'image generation', 'stable diffusion', 'dalle',
                'image synthesis', 'conditional generation'
            ],
            'Video Generation': [
                'text-to-video', 'video generation', 'video diffusion', 'motion generation'
            ],
            '3D Generation': [
                '3d generation', 'text-to-3d', 'point cloud', 'neural radiance field', 'nerf'
            ],
            'Diffusion Theory': [
                'sampling', 'guidance', 'classifier-free', 'score-based', 'denoising'
            ]
        }
    },
    'Time Series Analysis': {
        'keywords': [
            'time series', 'temporal', 'forecasting', 'prediction',
            'sequence modeling', 'time-series'
        ],
        'subdomains': {
            'Forecasting': [
                'time series forecasting', 'temporal prediction', 'demand forecasting',
                'traffic forecasting', 'financial forecasting'
            ],
            'Anomaly Detection': [
                'anomaly detection', 'outlier', 'time series anomaly', 'fault detection'
            ],
            'Classification': [
                'time series classification', 'segmentation', 'motif discovery'
            ]
        }
    }
}


@dataclass
class DomainPaper:
    """Paper with domain analysis metadata."""
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    # Analysis fields
    matched_keywords: List[str] = field(default_factory=list)
    subdomain: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class SubdomainAnalysis:
    """Analysis result for a subdomain."""
    name: str
    paper_count: int
    papers_with_abstract: int
    yearly_counts: Dict[int, int]
    top_keywords: Dict[str, int]
    representative_papers: List[Dict]
    emerging_trends: List[Tuple[str, float]]
    growth_rate: float


@dataclass
class DomainAnalysisReport:
    """Comprehensive domain analysis report."""
    domain: str
    year_range: Tuple[int, int]
    total_papers: int
    papers_with_abstract: int

    # Subdomain analysis
    subdomains: Dict[str, SubdomainAnalysis]

    # Cross-cutting analysis
    top_keywords: Dict[str, int]
    yearly_trends: Dict[int, int]

    # Venue distribution
    venue_distribution: Dict[str, int]

    # Top papers
    representative_papers: List[Dict]

    # Emerging topics
    emerging_topics: List[Tuple[str, float]]

    # Analytical insights
    insights: List[str]


class DeepDomainAnalyzer:
    """Deep analysis for specific research domains."""

    def __init__(self, data_dir: str = None):
        """
        Initialize analyzer.

        Args:
            data_dir: Path to paper data directory
        """
        if data_dir is None:
            root_dir = Path(__file__).parent.parent.parent.parent
            data_dir = root_dir / "data" / "paper"

        self.data_dir = Path(data_dir)
        self.domain_definitions = DOMAIN_DEFINITIONS

    def _parse_authors(self, authors_data) -> List[str]:
        """Parse authors from various formats."""
        if not authors_data:
            return []

        if isinstance(authors_data, str):
            return [authors_data]

        if isinstance(authors_data, dict):
            author_list = authors_data.get('author', [])
            if isinstance(author_list, list):
                return [a.get('text', '') for a in author_list if a.get('text')]
            elif isinstance(author_list, dict):
                return [author_list.get('text', '')]
            return []

        if isinstance(authors_data, list):
            return [a.get('text', '') for a in authors_data if isinstance(a, dict) and a.get('text')]

        return []

    def _load_papers(self, conferences: List[str], years: List[int]) -> List[DomainPaper]:
        """Load papers from specified conferences and years."""
        papers = []

        for conf_dir in self.data_dir.iterdir():
            if not conf_dir.is_dir() or 'conf' not in conf_dir.name.lower():
                continue

            for conf in conferences:
                for year in years:
                    json_file = conf_dir / f"{conf.lower()}_{year}.json"
                    if not json_file.exists():
                        continue

                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        for paper_dict in data.get('papers', []):
                            authors = self._parse_authors(paper_dict.get('authors', []))
                            paper = DomainPaper(
                                title=paper_dict.get('title', ''),
                                authors=authors,
                                year=year,
                                venue=paper_dict.get('venue', ''),
                                abstract=paper_dict.get('abstract'),
                                doi=paper_dict.get('doi'),
                                url=paper_dict.get('ee')
                            )
                            papers.append(paper)
                    except Exception as e:
                        logger.warning(f"Error loading {json_file}: {e}")

        return papers

    def _match_keywords(self, text: str, keywords: List[str]) -> Tuple[List[str], float]:
        """Match keywords in text and calculate relevance score."""
        text_lower = text.lower()
        matched = []
        match_count = 0

        for keyword in keywords:
            # Use word boundary matching for better accuracy
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(keyword)
                match_count += 1

        score = match_count / len(keywords) if keywords else 0
        return matched, score

    def _classify_subdomain(self, text: str, subdomains: Dict[str, List[str]]) -> Optional[str]:
        """Classify paper into a subdomain."""
        text_lower = text.lower()
        subdomain_scores = Counter()

        for subdomain, keywords in subdomains.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    subdomain_scores[subdomain] += 1

        if subdomain_scores:
            return subdomain_scores.most_common(1)[0][0]
        return None

    def analyze_domain(
        self,
        domain: str,
        conferences: List[str] = None,
        years: List[int] = None,
        min_relevance: float = 0.1,
        top_papers: int = 10
    ) -> DomainAnalysisReport:
        """
        Perform deep analysis for a specific domain.

        Args:
            domain: Domain name (e.g., 'AI Agent', 'Federated Learning')
            conferences: List of conferences to analyze
            years: List of years to analyze
            min_relevance: Minimum relevance score to include paper
            top_papers: Number of representative papers to include

        Returns:
            DomainAnalysisReport with comprehensive analysis
        """
        # Get domain definition
        domain_def = self.domain_definitions.get(domain)
        if not domain_def:
            # Try to find partial match
            for defined_domain in self.domain_definitions:
                if defined_domain.lower() in domain.lower() or domain.lower() in defined_domain.lower():
                    domain_def = self.domain_definitions[defined_domain]
                    domain = defined_domain
                    break

        if not domain_def:
            logger.warning(f"Domain '{domain}' not found in definitions. Using keyword matching only.")
            domain_def = {'keywords': domain.split(), 'subdomains': {}}

        keywords = domain_def.get('keywords', [])
        subdomains_def = domain_def.get('subdomains', {})

        # Default conferences and years
        if conferences is None:
            conferences = ['aaai', 'nips', 'iclr', 'icml', 'acl', 'cvpr', 'kdd', 'sigir']
        if years is None:
            years = [2023, 2024, 2025]

        # Load papers
        logger.info(f"Loading papers from {conferences} for years {years}")
        papers = self._load_papers(conferences, years)
        logger.info(f"Loaded {len(papers)} papers")

        # Filter papers by domain relevance
        domain_papers = []
        for paper in papers:
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract

            matched, score = self._match_keywords(text, keywords)
            if score >= min_relevance:
                paper.matched_keywords = matched
                paper.relevance_score = score

                # Classify subdomain
                if subdomains_def:
                    paper.subdomain = self._classify_subdomain(text, subdomains_def)

                domain_papers.append(paper)

        logger.info(f"Found {len(domain_papers)} papers relevant to {domain}")

        if not domain_papers:
            return DomainAnalysisReport(
                domain=domain,
                year_range=(min(years), max(years)),
                total_papers=0,
                papers_with_abstract=0,
                subdomains={},
                top_keywords={},
                yearly_trends={},
                venue_distribution={},
                representative_papers=[],
                emerging_topics=[],
                insights=["No papers found matching the domain criteria."]
            )

        # Analyze yearly trends
        yearly_counts = Counter(p.year for p in domain_papers)

        # Analyze venue distribution
        venue_dist = Counter(p.venue for p in domain_papers)

        # Analyze subdomains
        subdomain_papers = defaultdict(list)
        for paper in domain_papers:
            if paper.subdomain:
                subdomain_papers[paper.subdomain].append(paper)

        subdomain_analyses = {}
        for subdomain_name, subdomain_paper_list in subdomain_papers.items():
            analysis = self._analyze_subdomain(
                subdomain_name,
                subdomain_paper_list,
                top_papers
            )
            subdomain_analyses[subdomain_name] = analysis

        # Extract top keywords across all domain papers
        top_keywords = self._extract_keywords(domain_papers)

        # Get representative papers
        representative = self._get_representative_papers(domain_papers, top_papers)

        # Analyze emerging topics
        emerging = self._analyze_emerging_topics(domain_papers, years)

        # Generate insights
        insights = self._generate_insights(
            domain, domain_papers, yearly_counts, venue_dist,
            subdomain_analyses, emerging, years
        )

        return DomainAnalysisReport(
            domain=domain,
            year_range=(min(years), max(years)),
            total_papers=len(domain_papers),
            papers_with_abstract=sum(1 for p in domain_papers if p.abstract),
            subdomains=subdomain_analyses,
            top_keywords=top_keywords,
            yearly_trends=dict(yearly_counts),
            venue_distribution=dict(venue_dist),
            representative_papers=representative,
            emerging_topics=emerging,
            insights=insights
        )

    def _analyze_subdomain(self, name: str, papers: List[DomainPaper], top_papers: int) -> SubdomainAnalysis:
        """Analyze a specific subdomain."""
        yearly_counts = Counter(p.year for p in papers)

        # Calculate growth rate
        years = sorted(yearly_counts.keys())
        if len(years) >= 2:
            early = sum(yearly_counts[y] for y in years[:len(years)//2])
            late = sum(yearly_counts[y] for y in years[len(years)//2:])
            growth = (late - early) / early if early > 0 else 0
        else:
            growth = 0

        # Extract keywords
        top_keywords = self._extract_keywords(papers, limit=20)

        # Get representative papers
        representative = self._get_representative_papers(papers, top_papers)

        # Emerging trends
        emerging = self._analyze_emerging_topics(papers, sorted(yearly_counts.keys()))

        return SubdomainAnalysis(
            name=name,
            paper_count=len(papers),
            papers_with_abstract=sum(1 for p in papers if p.abstract),
            yearly_counts=dict(yearly_counts),
            top_keywords=top_keywords,
            representative_papers=representative,
            emerging_trends=emerging,
            growth_rate=growth
        )

    def _extract_keywords(self, papers: List[DomainPaper], limit: int = 30) -> Dict[str, int]:
        """Extract top keywords from papers."""
        word_counts = Counter()

        # Comprehensive stopwords - including:
        # 1. English common words
        # 2. Generic research paper words
        # 3. Auxiliary verbs and pronouns
        # 4. Quantifiers and determiners
        stopwords = {
            # Basic English stopwords
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'we', 'they', 'he', 'she', 'it', 'my', 'our', 'their', 'its', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'also', 'now', 'even', 'still',
            'both', 'either', 'neither', 'another', 'much', 'many', 'any', 'about',
            'like', 'get', 'got', 'go', 'going', 'went', 'gone', 'come', 'came',
            'make', 'made', 'take', 'took', 'taken', 'see', 'saw', 'seen', 'know',
            'knew', 'known', 'think', 'thought', 'want', 'use', 'used', 'using',
            'find', 'found', 'give', 'gave', 'given', 'tell', 'told', 'seem',
            'leave', 'put', 'keep', 'kept', 'let', 'begin', 'began', 'begun',
            'appear', 'said', 'say', 'says', 'one', 'two', 'three', 'four', 'five',

            # Generic research paper words (not informative)
            'paper', 'approach', 'method', 'methods', 'system', 'systems', 'result',
            'results', 'work', 'works', 'propose', 'proposed', 'show', 'shows',
            'shown', 'demonstrate', 'demonstrates', 'demonstrated', 'present',
            'presents', 'presented', 'introduce', 'introduces', 'introduced',
            'proposal', 'novel', 'new', 'different', 'various', 'several', 'many',
            'number', 'number of', 'based', 'using', 'however', 'although', 'though',
            'while', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
            'additionally', 'specifically', 'generally', 'typically', 'usually',
            'often', 'sometimes', 'always', 'never', 'ever', 'already', 'yet',
            'since', 'due', 'because', 'following', 'previous', 'prior', 'existing',
            'current', 'recent', 'recently', 'related', 'concerned', 'compared',
            'addition', 'well', 'good', 'better', 'best', 'high', 'higher', 'highest',
            'low', 'lower', 'large', 'larger', 'largest', 'small', 'smaller',
            'first', 'second', 'third', 'last', 'next', 'final', 'main', 'major',
            'significant', 'importance', 'important', 'able', 'unable', 'possible',
            'impossible', 'difficult', 'easy', 'simple', 'complex', 'different',
            'similar', 'order', 'require', 'requires', 'required', 'need', 'needs',
            'able', 'capable', 'perform', 'performs', 'performance', 'achieve',
            'achieves', 'achieved', 'obtain', 'obtains', 'obtained', 'provide',
            'provides', 'provided', 'enable', 'enables', 'applied', 'apply', 'applies',

            # Common AI/ML terms that are too generic in context
            'learning', 'model', 'models', 'training', 'train', 'test', 'testing',
            'data', 'dataset', 'datasets', 'feature', 'features', 'input', 'inputs',
            'output', 'outputs', 'prediction', 'predictions', 'accuracy', 'score',
            'scores', 'state', 'states', 'artificial', 'intelligence', 'deep',

            # Other common filler words
            'case', 'cases', 'point', 'points', 'term', 'terms', 'way', 'ways',
            'aspect', 'aspects', 'issue', 'issues', 'problem', 'problems',
            'question', 'questions', 'answer', 'answers', 'part', 'parts',
            'thing', 'things', 'stuff', 'area', 'areas', 'field', 'fields',
            'level', 'types', 'kind', 'form', 'forms', 'set', 'sets',
            'process', 'processes', 'type', 'value', 'values', 'time', 'times',
            'year', 'years', 'day', 'days', 'person', 'people', 'group', 'groups',
            'word', 'words', 'example', 'examples', 'regard', 'regards', 'without'
        }

        for paper in papers:
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract

            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            words = [w for w in words if w not in stopwords and len(w) > 2]

            # Count unique words per paper to avoid over-weighting
            unique_words = set(words)
            word_counts.update(unique_words)

        return dict(word_counts.most_common(limit))

    def _get_representative_papers(self, papers: List[DomainPaper], top_n: int) -> List[Dict]:
        """Get representative papers (highest relevance scores)."""
        # Sort by relevance score
        sorted_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)

        representative = []
        for paper in sorted_papers[:top_n]:
            representative.append({
                'title': paper.title,
                'authors': paper.authors[:5],  # Top 5 authors
                'year': paper.year,
                'venue': paper.venue,
                'relevance_score': paper.relevance_score,
                'matched_keywords': paper.matched_keywords[:5],
                'abstract': paper.abstract[:300] + '...' if paper.abstract and len(paper.abstract) > 300 else paper.abstract
            })

        return representative

    def _analyze_emerging_topics(self, papers: List[DomainPaper], years: List[int]) -> List[Tuple[str, float]]:
        """Analyze emerging topics based on growth rate."""
        if len(years) < 2:
            return []

        # Stopwords to filter out from emerging topics
        emerging_stopwords = {
            # Common English words
            'have', 'has', 'had', 'having', 'does', 'doing', 'done', 'make', 'made',
            'going', 'goes', 'went', 'gone', 'come', 'comes', 'came', 'take', 'takes',
            'took', 'taken', 'see', 'sees', 'saw', 'seen', 'know', 'knows', 'knew',
            'think', 'thinks', 'thought', 'want', 'wants', 'use', 'uses', 'using',
            'find', 'finds', 'found', 'give', 'gives', 'gave', 'given', 'tell', 'tells',
            'told', 'become', 'becomes', 'became', 'leave', 'leaves', 'left', 'put',
            'keep', 'keeps', 'kept', 'let', 'begin', 'begins', 'began', 'begun',
            'appear', 'appears', 'appeared', 'consider', 'considers', 'considered',
            'these', 'those', 'their', 'there', 'here', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
            'with', 'without', 'within', 'across', 'along', 'among', 'after',
            'before', 'behind', 'beside', 'besides', 'between', 'during', 'under',
            'until', 'upon', 'within', 'without', 'each', 'any', 'all', 'being',
            'that', 'this', 'then', 'than', 'them', 'they', 'its', 'also', 'data',
            'one', 'ones', 'two', 'three', 'ensures', 'ensure', 'ensuring',
            'inherent', 'inherently', 'possess', 'possesses', 'possessing',
            # Common fillers
            'often', 'sometimes', 'usually', 'always', 'never', 'rarely',
            'however', 'although', 'though', 'while', 'therefore', 'thus',
            'hence', 'moreover', 'furthermore', 'additionally', 'specifically',
            'generally', 'typically', 'particularly', 'essentially', 'basically',
            'actually', 'really', 'quite', 'rather', 'somewhat', 'completely',
            'entirely', 'especially', 'probably', 'possibly',
            'certainly', 'clearly', 'obviously', 'seems', 'appears', 'may',
            'might', 'could', 'would', 'should', 'must', 'need', 'needs',
            'recent', 'recently', 'current', 'existing', 'proposed', 'novel',
            'generating', 'generated', 'generation', 'driven', 'drive',
            'when', 'where', 'what', 'which', 'who', 'how', 'process',
            'simply', 'nicely', 'successfully', 'effectively', 'efficiently',
            'significantly', 'dramatically', 'substantially', 'gradually',
            'rapidly', 'quickly', 'slowly', 'suddenly', 'ultimately',
            'specialized', 'specialize', 'generate', 'generate', 'generated',
            'high', 'higher', 'low', 'lower', 'large', 'larger', 'small',
            'better', 'best', 'worse', 'worst', 'good', 'bad', 'well',
            'potential', 'potentially', 'successfully', 'success', 'effectiveness',
            # Common AI/ML terms (too generic in this context)
            'models', 'model', 'human', 'humans', 'human-like', 'human-level',
            'state', 'states', 'space', 'spaces', 'sample', 'samples', 'sampling',
            'real', 'world', 'real-world', 'realistic', 'synthetic', 'public',
            'private', 'first', 'second', 'third', 'next', 'last', 'final',
            # Generic research terms
            'based', 'task', 'tasks', 'challenges', 'challenging', 'approach', 'method', 'methods',
            'problem', 'problems', 'issue', 'issues', 'solution', 'solutions',
            'results', 'experiments', 'experiment', 'performance', 'accuracy',
            'show', 'shows', 'shown', 'demonstrate', 'demonstrates', 'propose',
            'proposed', 'paper', 'work', 'research', 'study', 'paper', 'novel',
            'various', 'different', 'several', 'multiple', 'number', 'case',
            'cases', 'example', 'examples', 'regard', 'regards', 'aspect',
            'aspects', 'thing', 'things', 'stuff', 'point', 'way', 'ways',
            'design', 'designed', 'designing', 'designs', 'framework', 'frameworks',
            # Vague terms
            'enable', 'enables', 'enhanced', 'enhancing', 'enhances',
            'improve', 'improved', 'improving', 'improvement',
            'address', 'addresses', 'addressing', 'addressed',
            'introduce', 'introduces', 'introduced', 'introduction',
            'present', 'presents', 'presented', 'presenting',
            'provide', 'provides', 'provided', 'providing',
            'obtain', 'obtains', 'obtained', 'obtaining',
            'achieve', 'achieves', 'achieved', 'achieving',
            'propose', 'proposes', 'proposed', 'proposing',
            'capability', 'capabilities', 'capable', 'ability', 'abilities',
            'leads', 'lead', 'leading', 'results', 'result', 'following', 'follows'
        }

        # Extract keywords by year
        yearly_keywords = defaultdict(Counter)
        for paper in papers:
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract

            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            # Filter stopwords
            words = [w for w in words if w not in emerging_stopwords]
            yearly_keywords[paper.year].update(set(words))

        # Calculate growth rates
        mid_point = len(years) // 2
        early_years = years[:mid_point]
        late_years = years[mid_point:]

        topic_growth = []
        all_keywords = set()
        for year in years:
            all_keywords.update(yearly_keywords[year].keys())

        for keyword in all_keywords:
            early_count = sum(yearly_keywords[y].get(keyword, 0) for y in early_years)
            late_count = sum(yearly_keywords[y].get(keyword, 0) for y in late_years)

            if early_count >= 5:  # Minimum threshold increased to 5
                growth = (late_count - early_count) / early_count
                if growth > 0.5:  # At least 50% growth
                    topic_growth.append((keyword, growth))

        topic_growth.sort(key=lambda x: x[1], reverse=True)
        return topic_growth[:15]

    def _generate_insights(
        self,
        domain: str,
        papers: List[DomainPaper],
        yearly_counts: Counter,
        venue_dist: Counter,
        subdomain_analyses: Dict[str, SubdomainAnalysis],
        emerging: List[Tuple[str, float]],
        years: List[int]
    ) -> List[str]:
        """Generate analytical insights."""
        insights = []

        # Overall trend insight
        if len(years) >= 2:
            sorted_years = sorted(yearly_counts.keys())
            first_year = yearly_counts[sorted_years[0]]
            last_year = yearly_counts[sorted_years[-1]]
            total_growth = (last_year - first_year) / first_year if first_year > 0 else 0

            if total_growth > 0.5:
                insights.append(f"📈 {domain} is experiencing rapid growth, with paper count increasing {total_growth*100:.1f}% from {sorted_years[0]} to {sorted_years[-1]}.")
            elif total_growth > 0:
                insights.append(f"📊 {domain} shows steady growth, increasing by {total_growth*100:.1f}% over the period.")
            else:
                insights.append(f"📉 {domain} paper count has remained relatively stable.")

        # Top venues insight
        if venue_dist:
            top_venue = venue_dist.most_common(1)[0]
            insights.append(f"🎯 {top_venue[0]} is the most productive venue for {domain} papers ({top_venue[1]} papers).")

        # Subdomain insights
        if subdomain_analyses:
            # Find fastest growing subdomain
            growing_subdomains = [(name, analysis.growth_rate) for name, analysis in subdomain_analyses.items()]
            growing_subdomains.sort(key=lambda x: x[1], reverse=True)

            if growing_subdomains and growing_subdomains[0][1] > 0.3:
                insights.append(f"🚀 '{growing_subdomains[0][0]}' is the fastest growing subdirection with {growing_subdomains[0][1]*100:.1f}% growth.")

            # Largest subdomain
            largest = max(subdomain_analyses.items(), key=lambda x: x[1].paper_count)
            insights.append(f"📚 '{largest[0]}' is the largest subdirection with {largest[1].paper_count} papers.")

        # Emerging topics insight
        if emerging:
            top_emerging = emerging[:3]
            emerging_str = ', '.join([f"'{t[0]}'" for t in top_emerging])
            insights.append(f"🔮 Emerging topics include: {emerging_str}.")

        # Abstract coverage
        with_abstract = sum(1 for p in papers if p.abstract)
        coverage = with_abstract / len(papers) * 100 if papers else 0
        insights.append(f"📄 {coverage:.1f}% of papers have abstracts available for deeper analysis.")

        return insights


def format_report(report: DomainAnalysisReport) -> str:
    """Format analysis report as readable text."""
    output = []
    output.append("=" * 80)
    output.append(f"📊 Deep Domain Analysis: {report.domain}")
    output.append("=" * 80)

    # Overview
    output.append(f"\n📅 Year Range: {report.year_range[0]} - {report.year_range[1]}")
    output.append(f"📄 Total Papers: {report.total_papers}")
    output.append(f"📝 Papers with Abstract: {report.papers_with_abstract}")

    # Insights
    output.append("\n" + "=" * 80)
    output.append("💡 Key Insights")
    output.append("=" * 80)
    for insight in report.insights:
        output.append(f"  {insight}")

    # Yearly trends
    output.append("\n" + "=" * 80)
    output.append("📈 Yearly Trends")
    output.append("=" * 80)
    for year in sorted(report.yearly_trends.keys()):
        count = report.yearly_trends[year]
        bar = "█" * (count // 5)
        output.append(f"  {year}: {count:4d} {bar}")

    # Venue distribution
    output.append("\n" + "=" * 80)
    output.append("🏛️ Venue Distribution (Top 5)")
    output.append("=" * 80)
    sorted_venues = sorted(report.venue_distribution.items(), key=lambda x: x[1], reverse=True)
    for venue, count in sorted_venues[:5]:
        pct = count / report.total_papers * 100
        output.append(f"  {venue}: {count} ({pct:.1f}%)")

    # Subdomain analysis
    if report.subdomains:
        output.append("\n" + "=" * 80)
        output.append("🎯 Subdomain Analysis")
        output.append("=" * 80)

        # Sort by paper count
        sorted_subdomains = sorted(
            report.subdomains.items(),
            key=lambda x: x[1].paper_count,
            reverse=True
        )

        for subdomain_name, analysis in sorted_subdomains:
            output.append(f"\n  ▶ {subdomain_name}")
            output.append(f"    Papers: {analysis.paper_count}")
            output.append(f"    Growth: {analysis.growth_rate*100:+.1f}%")

            # Yearly breakdown
            years_str = ", ".join([f"{y}:{c}" for y, c in sorted(analysis.yearly_counts.items())])
            output.append(f"    Timeline: {years_str}")

            # Top keywords
            if analysis.top_keywords:
                top_kws = list(analysis.top_keywords.keys())[:8]
                output.append(f"    Keywords: {', '.join(top_kws)}")

    # Emerging topics
    if report.emerging_topics:
        output.append("\n" + "=" * 80)
        output.append("🔮 Emerging Topics")
        output.append("=" * 80)
        for topic, growth in report.emerging_topics[:10]:
            output.append(f"  {topic}: +{growth*100:.1f}% growth")

    # Representative papers
    if report.representative_papers:
        output.append("\n" + "=" * 80)
        output.append("📚 Representative Papers (Top 5)")
        output.append("=" * 80)

        for i, paper in enumerate(report.representative_papers[:5], 1):
            output.append(f"\n  [{i}] {paper['title'][:70]}...")
            output.append(f"      {paper['authors'][0]} et al. | {paper['year']} | {paper['venue']}")
            output.append(f"      Matched: {', '.join(paper['matched_keywords'][:3])}")

    output.append("\n" + "=" * 80)

    return "\n".join(output)


def analyze_vocabulary_turnover(
    papers: List,
    years: List[int],
    top_n: int = 50
) -> Dict:
    """
    分析领域词汇的新陈代谢 - 追踪词汇的兴衰

    Returns:
        {
            'rising_keywords': [{'word': str, 'change': float}],
            'declining_keywords': [{'word': str, 'change': float}],
            'yearly_top_keywords': {year: [(word, count)]}
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
    all_words = set(w for year_words in word_by_year.values() for w in year_words)

    for word in all_words:
        counts = [word_by_year.get(y, {}).get(word, 0) for y in sorted(years)]

        if sum(counts) < 5:  # 过滤太低频
            continue

        early = np.mean(counts[:2]) if len(counts) >= 2 else counts[0]
        late = np.mean(counts[-2:]) if len(counts) >= 2 else counts[-1]

        if early > 0:
            change = (late - early) / early
            keyword_trends.append({
                'word': word,
                'change': round(change, 3),
                'early_avg': round(early, 1),
                'late_avg': round(late, 1)
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


if __name__ == "__main__":
    # Test domain analysis
    analyzer = DeepDomainAnalyzer()

    # Analyze AI Agent domain
    print("Analyzing AI Agent domain...")
    report = analyzer.analyze_domain(
        domain="AI Agent",
        years=[2023, 2024, 2025]
    )

    print(format_report(report))
