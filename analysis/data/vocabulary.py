"""
Vocabulary management for paper analysis.
Contains stopwords to filter and synonym mappings to merge.
"""

# =============================================================================
# Stopwords - Words to filter out during analysis
# These are common words that don't carry meaningful research information
# Based on analysis of 33,436 CCF-A papers (2015-2025)
# =============================================================================

STOPWORDS = {
    # Common English words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'been', 'being',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    'there', 'then', 'once', 'if', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'while', 'own', 'out', 'off', 'over', 'any', 'because',
    'until', 'against', 'among', 'yet', 'even', 'still', 'well', 'back',

    # Additional pronouns and possessive adjectives (from analysis)
    'our', 'our', 'our', 'our', 'our',  # Repeated for emphasis - very high freq
    'their', 'they', 'them', 'his', 'her', 'its', 'my', 'your', 'itself',
    'ours', 'theirs', 'him', 'hers', 'myself', 'yourself', 'themselves',

    # Additional verbs (from analysis)
    'are', 'is', 'was', 'were', 'being', 'been', 'am', "isn't", "aren't",
    "wasn't", "weren't", 'does', 'did', 'doing', "don't", "doesn't", "didn't",
    'having', 'had', 'has',

    # Academic connectors and transitions (from analysis - high frequency)
    'however', 'via', 'often', 'due', 'without', 'compared', 'specifically',
    'finally', 'thus', 'hence', 'therefore', 'moreover', 'furthermore',
    'additionally', 'although', 'whereas', 'since', 'because', 'namely',
    'especially', 'particularly', 'typically', 'usually', 'generally',
    'actually', 'already', 'always', 'never', 'sometimes', 'perhaps',
    'possibly', 'probably', 'likely', 'unlikely', 'essentially', 'basically',
    'approximately', 'exactly', 'recently', 'currently', 'previously',
    'similarly', 'differently', 'alternatively', 'consequently', 'accordingly',

    # Academic writing common words
    'paper', 'work', 'show', 'showed', 'shown', 'shows', 'propose', 'proposed',
    'proposes', 'present', 'presented', 'presents', 'presenting', 'use', 'used',
    'using', 'uses', 'approach', 'method', 'methods', 'result', 'results',
    'based', 'using', 'however', 'therefore', 'thus', 'hence', 'moreover',
    'furthermore', 'additionally', 'although', 'whereas', 'since', 'because',
    'given', 'consider', 'considered', 'considering', 'assume', 'assumed',
    'assumption', 'define', 'defined', 'definition', 'denote', 'denoted',
    'denotes', 'refer', 'referred', 'refers', 'describe', 'described',
    'describes', 'describing', 'obtain', 'obtained', 'obtains', 'obtain',
    'given', 'provides', 'provide', 'provided', 'provides', 'also', 'new',
    'novel', 'different', 'similar', 'existing', 'current', 'previous',
    'prior', 'earlier', 'recent', 'related', 'various', 'many', 'several',
    'specific', 'general', 'specific', 'particular', 'certain', 'several',

    # Technical but meaningless in analysis (high frequency from analysis)
    'study', 'studies', 'research', 'analysis', 'analysises', 'problem',
    'problems', 'solution', 'solutions', 'model', 'models', 'system', 'systems',
    'framework', 'frameworks', 'algorithm', 'algorithms', 'technique', 'techniques',
    'task', 'tasks', 'data', 'dataset', 'datasets', 'performance', 'accuracy',
    'learning', 'training', 'trained', 'test', 'testing', 'tested',
    'experiments', 'experiment', 'evaluation', 'evaluations',
    'benchmark', 'benchmarks', 'baseline', 'baselines',
    'demonstrate', 'demonstrated', 'demonstrates', 'demonstrating',
    'showed', 'shows', 'demonstrate', 'obtain', 'obtained', 'obtains',
    'provide', 'provides', 'provided', 'providing',
    'propose', 'proposed', 'proposes', 'proposing',
    'address', 'addresses', 'addressed', 'addressing',
    'introduce', 'introduces', 'introduced', 'introduction',
    'achieve', 'achieves', 'achieved', 'achieving',
    'improve', 'improved', 'improving', 'improvement',
    'outperforms', 'effective', 'efficiency', 'efficient',
    'significant', 'significantly', 'importance', 'important',
    'relevant', 'useful', 'valuable', 'useful',

    # Numbers and measurements (too generic)
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'first', 'second', 'third', 'fourth', 'fifth', 'many', 'much',
    'less', 'least', 'greater', 'greatest', 'high', 'higher', 'highest',
    'low', 'lower', 'best', 'better', 'good', 'well', 'bad', 'worse', 'worst',
    'large', 'larger', 'largest', 'small', 'smaller', 'smallest', 'big',
    'bigger', 'biggest', 'long', 'longer', 'longest', 'short', 'shorter',
    'shortest', 'wide', 'wider', 'widest', 'deep', 'deeper', 'deepest',

    # Common adjectives
    'real', 'true', 'false', 'possible', 'impossible', 'necessary', 'sufficient',
    'interesting', 'popular', 'common', 'rare', 'unknown', 'available', 'feasible',
    'practical', 'theoretical', 'empirical', 'experimental', 'quantitative', 'qualitative',
    'extensive', 'various', 'multiple', 'single', 'specific', 'general',

    # Common verbs
    'can', 'may', 'able', 'allow', 'allows', 'enable', 'enables', 'help',
    'helps', 'make', 'makes', 'made', 'get', 'gets', 'getting', 'got',
    'take', 'takes', 'took', 'taken', 'give', 'gives', 'gave', 'given',
    'let', 'lets', 'put', 'puts', 'set', 'sets', 'run', 'runs', 'ran',
    'going', 'goes', 'went', 'come', 'comes', 'came', 'see', 'sees', 'saw',
    'seen', 'know', 'knows', 'knew', 'known', 'think', 'thinks', 'thought',
    'want', 'wants', 'wanted', 'like', 'likes', 'need', 'needs', 'needed',
    'over', 'across', 'given', 'across',

    # URL fragments and artifacts (from paper scraping)
    'https', 'http', 'www', 'com', 'org', 'net', 'edu', 'pdf', 'arxiv',
    'doi', 'url', 'html', 'htm', 'github',

    # Single characters and meaningless
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',

    # Common research artifacts
    'table', 'figure', 'fig', 'section', 'chapter', 'page', 'vol', 'issue',
    'conference', 'journal', 'proceedings', 'acm', 'ieee', 'arxiv', 'preprint',

    # Additional high-frequency but uninformative terms from analysis
    'state', 'art', 'large', 'knowledge', 'image', 'images', 'text', 'graph',
    'domain', 'human', 'features', 'feature', 'semantic', 'detection',
    'reasoning', 'representation', 'representations', 'visual', 'classification',
    'prediction', 'translation', 'recognition', 'optimization', 'inference',
    'generation', 'generation', 'label', 'labels', 'labeling',
    'source', 'sources', 'target', 'targets', 'question', 'questions',
    'user', 'users', 'quality', 'space', 'temporal', 'real', 'world',
    'level', 'set', 'sets', 'cross', 'context', 'fine', 'scale', 'scales',
    'address', 'design', 'applications', 'understanding', 'available',

    # More analysis artifacts (additional from execution log)
    'effectiveness', 'effectively', 'performance', 'real-world', 'realworld',
    'proposed', 'proposing', 'existing', 'various', 'wide', 'range', 'ranges',
    'able', 'neither', 'either', 'despite', 'although', 'whereas', 'whether',
    'overall', 'particular', 'overall', 'per', 'vs', 'versus', 'eg', 'ie',
    'etc', 'et', 'al', 'cf', 'ref', 'refs', 'see', 'cf', 'vs', 'within',
    'along', 'across', 'around', 'among', 'throughout', 'onto', 'upon',
    'above', 'below', 'beyond', 'near', 'besides', 'beside',

    # Additional generic terms from recent analysis
    'stateoftheart', 'state_of_the_art', 'sota',  # Common but generic
    'challenging', 'challenges', 'challenge',
    'aims', 'aim', 'goal', 'goals', 'objective', 'objectives',
    'process', 'processes', 'processing',
    'module', 'modules', 'component', 'components',
    'strategy', 'strategies', 'approach',  # Too generic
    'including', 'included', 'include', 'includes',
    'generate', 'generated', 'generates', 'generation',
    'distribution', 'distributions', 'distributed',
    'information', 'regarding', 'concerning',
    'terms', 'conditions', 'respect', 'instance',
    'potential', 'potentially', 'significantly',
}

# =============================================================================
# Synonym mappings - Words to merge during analysis
# Maps variants to canonical forms
# Based on bigram analysis of CCF-A papers
# =============================================================================

SYNONYMS = {
    # Machine Learning / AI
    'machine learning': 'machine_learning',
    'machine-learning': 'machine_learning',
    'deep learning': 'deep_learning',
    'deep-learning': 'deep_learning',
    'neural network': 'neural_network',
    'neural networks': 'neural_network',
    'neural-net': 'neural_network',
    'nn': 'neural_network',
    'nns': 'neural_network',

    # NLP
    'natural language processing': 'nlp',
    'natural-language-processing': 'nlp',
    'natural language': 'nlp',
    'language model': 'language_model',
    'language models': 'language_model',
    'language_model': 'language_model',
    'large language model': 'llm',
    'large language models': 'llm',
    'large-language-model': 'llm',
    'llm': 'llm',
    'llms': 'llm',

    # Computer Vision
    'computer vision': 'computer_vision',
    'computer-vision': 'computer_vision',
    'cv': 'computer_vision',
    'convolutional neural network': 'cnn',
    'convolutional neural networks': 'cnn',
    'convolutional neural net': 'cnn',
    'cnn': 'cnn',
    'vision transformer': 'vit',
    'vit': 'vit',

    # Reinforcement Learning
    'reinforcement learning': 'rl',
    'reinforcement-learning': 'rl',
    'rl': 'rl',

    # Deep Reinforcement Learning
    'deep reinforcement learning': 'drl',
    'deep-reinforcement-learning': 'drl',
    'drl': 'drl',

    # Model types
    'generative adversarial network': 'gan',
    'generative adversarial networks': 'gan',
    'gans': 'gan',
    'vae': 'vae',
    'variational autoencoder': 'vae',
    'variational autoencoders': 'vae',
    'transformer': 'transformer',
    'transformers': 'transformer',

    # Training concepts
    'training': 'train',
    'trained': 'train',
    'train': 'train',
    'testing': 'test',
    'tested': 'test',
    'validation': 'valid',
    'valid': 'valid',
    'fine-tuning': 'fine_tune',
    'fine tuning': 'fine_tune',
    'fine_tune': 'fine_tune',
    'pre-training': 'pretrain',
    'pre training': 'pretrain',
    'pretrain': 'pretrain',
    'pre-trained': 'pretrain',

    # Evaluation metrics
    'accuracy': 'accuracy',
    'accuracies': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1_score',
    'f1_score': 'f1_score',
    'auc': 'auc',

    # Common research terms
    'experiment': 'experiment',
    'experiments': 'experiment',
    'experimental': 'experiment',
    'evaluation': 'evaluate',
    'evaluations': 'evaluate',
    'benchmark': 'benchmark',
    'benchmarks': 'benchmark',
    'state-of-the-art': 'sota',
    'state of the art': 'sota',
    'stateoftheart': 'sota',
    'state_art': 'sota',
    'sota': 'sota',
    'baseline': 'baseline',
    'baselines': 'baseline',

    # Graph / Network
    'graph neural network': 'gnn',
    'graph neural networks': 'gnn',
    'gnn': 'gnn',
    'gnns': 'gnn',
    'graph convolution': 'gc',
    'gcn': 'gcn',

    # Optimization
    'stochastic gradient descent': 'sgd',
    'sgd': 'sgd',
    'adam': 'adam',
    'adamw': 'adamw',

    # Attention mechanisms
    'self-attention': 'self_attention',
    'self attention': 'self_attention',
    'selfattention': 'self_attention',
    'cross-attention': 'cross_attention',
    'cross attention': 'cross_attention',

    # Multi-agent / RL
    'multi-agent': 'multi_agent',
    'multi agent': 'multi_agent',
    'multiagent': 'multi_agent',
    'multi agent systems': 'multi_agent',
    'agent': 'agent',
    'agents': 'agent',

    # Preprocessing
    'preprocessing': 'preprocess',
    'pre-processing': 'preprocess',
    'preprocess': 'preprocess',
    'tokenization': 'tokenize',
    'tokenizing': 'tokenize',
    'tokenize': 'tokenize',

    # Data augmentation
    'data augmentation': 'data_augmentation',
    'data-augmentation': 'data_augmentation',
    'augmentation': 'augmentation',
    'augment': 'augment',

    # Embedding
    'embedding': 'embedding',
    'embeddings': 'embedding',
    'word embedding': 'word_embedding',
    'word2vec': 'word2vec',
    'bert': 'bert',

    # Multi-modal
    'multi-modal': 'multimodal',
    'multi modal': 'multimodal',
    'multimodal': 'multimodal',
    'vision language': 'vision_language',
    'vision-language': 'vision_language',

    # Domain adaptation
    'domain adaptation': 'domain_adaptation',
    'domain-adaptation': 'domain_adaptation',

    # Zero-shot
    'zero-shot': 'zero_shot',
    'zero shot': 'zero_shot',
    'zero_shot': 'zero_shot',

    # Few-shot
    'few-shot': 'few_shot',
    'few shot': 'few_shot',
    'few_shot': 'few_shot',

    # Self-supervised
    'self-supervised': 'self_supervised',
    'self supervised': 'self_supervised',
    'self_supervised': 'self_supervised',

    # Semi-supervised
    'semi-supervised': 'semi_supervised',
    'semi supervised': 'semi_supervised',
    'semi_supervised': 'semi_supervised',

    # Supervised
    'supervised': 'supervised',
    'unsupervised': 'unsupervised',

    # Contrastive learning
    'contrastive learning': 'contrastive_learning',
    'contrastive-learning': 'contrastive_learning',

    # Question answering
    'question answering': 'question_answering',
    'question-answering': 'question_answering',

    # Machine translation
    'machine translation': 'machine_translation',
    'machine-translation': 'machine_translation',
}

# =============================================================================
# Singular/Plural mappings - Normalize word forms
# =============================================================================

SINGULAR_PLURAL = {
    # Common research terms - singular forms (plurals map to these)
    'models': 'model',
    'networks': 'network',
    'tasks': 'task',
    'methods': 'method',
    'approaches': 'approach',
    'systems': 'system',
    'algorithms': 'algorithm',
    'datasets': 'dataset',
    'features': 'feature',
    'images': 'image',
    'videos': 'video',
    'texts': 'text',
    'papers': 'paper',
    'results': 'result',
    'experiments': 'experiment',
    'problems': 'problem',
    'solutions': 'solution',
    'words': 'word',
    'sentences': 'sentence',
    'documents': 'document',
    'samples': 'sample',
    'examples': 'example',
    'classes': 'class',
    'labels': 'label',
    'categories': 'category',
    'groups': 'group',
    'users': 'user',
    'queries': 'query',
    'answers': 'answer',
    'questions': 'question',
    'representations': 'representation',
    'embeddings': 'embedding',
    'parameters': 'parameter',
    'layers': 'layer',
    'nodes': 'node',
    'edges': 'edge',
    'graphs': 'graph',
    'agents': 'agent',
    'policies': 'policy',
    'states': 'state',
    'actions': 'action',
    'rewards': 'reward',
    'losses': 'loss',
    'weights': 'weight',
    'biases': 'bias',
    'attacks': 'attack',
    'defenses': 'defense',
    'domains': 'domain',
    'benchmarks': 'benchmark',
    'baselines': 'baseline',
    'metrics': 'metric',
    'scores': 'score',
    'values': 'value',
    'ranks': 'rank',
    'vectors': 'vector',
    'matrices': 'matrix',
    'tensors': 'tensor',
    'dimensions': 'dimension',
    'spaces': 'space',
    'sources': 'source',
    'targets': 'target',
    'outputs': 'output',
    'inputs': 'input',
    'errors': 'error',
    'rates': 'rate',
    'probabilities': 'probability',
    'distributions': 'distribution',
    'instances': 'instance',
    'tokens': 'token',
    'tokenizers': 'tokenizer',
    'frameworks': 'framework',
    'techniques': 'technique',
    'studies': 'study',
    'analyses': 'analysis',
    'methodologies': 'methodology',
    'applications': 'application',
    'implementations': 'implementation',
    'architectures': 'architecture',
    'paradigms': 'paradigm',
    'generations': 'generation',
    'predictions': 'prediction',
    'classifications': 'classification',
    'recognitions': 'recognition',
    'detections': 'detection',
    'segmentations': 'segmentation',
    'retrievals': 'retrieval',
    'rankings': 'ranking',
    'searches': 'search',
    'translations': 'translation',
    'understandings': 'understanding',
    'reasonings': 'reasoning',
    'inferences': 'inference',
    'optimizations': 'optimization',
    'augmentations': 'augmentation',
    'preprocessings': 'preprocessing',
    'comparisons': 'comparison',
    'combinations': 'combination',
    'transformations': 'transformation',
    'contexts': 'context',
    'environments': 'environment',
    'scenarios': 'scenario',
    'settings': 'setting',
    'conditions': 'condition',
    'constraints': 'constraint',
    'objectives': 'objective',
    'functions': 'function',
    'properties': 'property',
    'attributes': 'attribute',
    'relationships': 'relationship',
    'connections': 'connection',
    'components': 'component',
    'modules': 'module',
    'units': 'unit',
    'blocks': 'block',
    'elements': 'element',
    'items': 'item',
    'patterns': 'pattern',
    'structures': 'structure',
    'hierarchies': 'hierarchy',
    'sequences': 'sequence',
    'streams': 'stream',
    'chains': 'chain',
    'paths': 'path',
    'routes': 'route',
    'steps': 'step',
    'stages': 'stage',
    'phases': 'phase',
    'levels': 'level',
    'degrees': 'degree',
    'scales': 'scale',
    'magnitudes': 'magnitude',
    'ranges': 'range',
    'intervals': 'interval',
    'periods': 'period',
    'durations': 'duration',
    'frequencies': 'frequency',
    'ratios': 'ratio',
    'percentages': 'percentage',
    'proportions': 'proportion',
    'quantities': 'quantity',
    'numbers': 'number',
    'counts': 'count',
    'totals': 'total',
    'sums': 'sum',
    'averages': 'average',
    'means': 'mean',
    'medians': 'median',
    'modes': 'mode',
    'variances': 'variance',
    'standards': 'standard',
}

# Merge singular/plural into main SYNONYMS
SYNONYMS.update(SINGULAR_PLURAL)

# =============================================================================
# Domain-specific vocabulary for CCF conferences
# =============================================================================

DOMAIN_VOCABULARY = {
    # AI/ML conferences (AAAI, IJCAI, NeurIPS, ICML, ICLR)
    'ai': ['artificial_intelligence', 'ai'],
    'ml': ['machine_learning', 'deep_learning', 'neural_network'],
    'nlp': ['nlp', 'natural_language_processing', 'language_model', 'text'],
    'cv': ['computer_vision', 'image', 'video', 'object_detection'],
    'rl': ['reinforcement_learning', 'rl', 'agent', 'policy'],
    'gnn': ['graph', 'gnn', 'graph_neural_network'],

    # Database (SIGMOD, VLDB, ICDE)
    'database': ['database', 'db', 'sql', 'query', 'transaction'],
    'data_management': ['data_management', 'data_processing', 'etl'],

    # IR/SIGIR
    'ir': ['information_retrieval', 'search', 'ranking', 'recommender'],
    'recommender': ['recommendation', 'recommender_system', 'collaborative_filtering'],

    # KDD
    'data_mining': ['data_mining', 'knowledge_discovery', 'analytics'],
    'mining': ['mining', 'pattern', 'association'],

    # MM
    'multimedia': ['multimedia', 'video', 'audio', 'image_retrieval'],

    # Security
    'security': ['security', 'privacy', 'cryptography', 'attack', 'defense'],
}


def get_stopwords() -> set:
    """Return the set of stopwords."""
    return STOPWORDS.copy()


def get_synonyms() -> dict:
    """Return the synonym mapping dictionary."""
    return SYNONYMS.copy()


def normalize_word(word: str) -> str:
    """
    Normalize a word by applying synonym mappings and lowercasing.

    Args:
        word: Word to normalize

    Returns:
        Normalized word
    """
    word_lower = word.lower()

    # Check direct synonym mapping
    if word_lower in SYNONYMS:
        return SYNONYMS[word_lower]

    # Check with underscores
    word_underscore = word_lower.replace('-', '_')
    if word_underscore in SYNONYMS:
        return SYNONYMS[word_underscore]

    return word_lower


def is_stopword(word: str) -> bool:
    """
    Check if a word is a stopword.

    Args:
        word: Word to check

    Returns:
        True if stopword, False otherwise
    """
    return word.lower() in STOPWORDS


def filter_words(words: list) -> list:
    """
    Filter out stopwords from a list of words.

    Args:
        words: List of words

    Returns:
        Filtered list without stopwords
    """
    return [w for w in words if not is_stopword(w)]


def normalize_words(words: list) -> list:
    """
    Normalize a list of words (lowercase + synonym mapping).

    Args:
        words: List of words

    Returns:
        List of normalized words (duplicates may exist)
    """
    return [normalize_word(w) for w in words]


def deduplicate_words(words: list) -> list:
    """
    Normalize and remove duplicates from word list.

    Args:
        words: List of words

    Returns:
        Deduplicated list of normalized words
    """
    normalized = normalize_words(words)
    return list(dict.fromkeys(normalized))  # Preserve order
