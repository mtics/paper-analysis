"""
Vocabulary management for paper analysis.
Contains stopwords to filter and synonym mappings to merge.
"""

# =============================================================================
# Stopwords - Words to filter out during analysis
# These are common words that don't carry meaningful research information
# =============================================================================

STOPWORDS = {
    # Common English words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
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

    # Technical but meaningless in analysis
    'study', 'studies', 'research', 'analysis', 'analysises', 'problem',
    'problems', 'solution', 'solutions', 'model', 'models', 'system', 'systems',
    'framework', 'frameworks', 'algorithm', 'algorithms', 'technique', 'techniques',
    'task', 'tasks', 'data', 'dataset', 'datasets', 'performance', 'accuracy',

    # Common verbs in papers
    'can', 'may', 'able', 'allow', 'allows', 'enable', 'enables', 'help',
    'helps', 'make', 'makes', 'made', 'get', 'gets', 'getting', 'got',
    'take', 'takes', 'took', 'taken', 'give', 'gives', 'gave', 'given',
    'let', 'lets', 'put', 'puts', 'set', 'sets', 'run', 'runs', 'ran',
    'going', 'goes', 'went', 'come', 'comes', 'came', 'see', 'sees', 'saw',
    'seen', 'know', 'knows', 'knew', 'known', 'think', 'thinks', 'thought',
    'want', 'wants', 'wanted', 'like', 'likes', 'need', 'needs', 'needed',

    # Numbers and measurements
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'first', 'second', 'third', 'fourth', 'fifth', 'many', 'much',
    'less', 'least', 'greater', 'greatest', 'high', 'higher', 'highest',
    'low', 'lower', 'best', 'better', 'good', 'well', 'bad', 'worse', 'worst',
    'large', 'larger', 'largest', 'small', 'smaller', 'smallest', 'big',
    ' bigger', 'biggest', 'long', 'longer', 'longest', 'short', 'shorter',
    'shortest', 'wide', 'wider', 'widest', 'deep', 'deeper', 'deepest',

    # Common adjectives
    'real', 'true', 'false', 'possible', 'impossible', 'necessary', 'sufficient',
    'important', 'significant', 'interesting', 'relevant', 'useful', 'valuable',
    'effective', 'efficient', 'simple', 'complex', 'difficult', 'easy',
    'fast', 'quick', 'slow', 'cheap', 'expensive', 'popular', 'common',
    'rare', 'unknown', 'available', 'feasible', 'practical', 'theoretical',
    'empirical', 'experimental', 'quantitative', 'qualitative',

    # URL fragments and artifacts
    'https', 'http', 'www', 'com', 'org', 'net', 'edu', 'pdf', 'arxiv',
    'doi', 'url', 'html', 'htm',

    # Single characters and meaningless
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',

    # Common research artifacts
    'table', 'figure', 'fig', 'section', 'chapter', 'page', 'vol', 'issue',
    'conference', 'journal', 'proceedings', 'acm', 'ieee', 'arxiv', 'preprint',
}

# =============================================================================
# Synonym mappings - Words to merge during analysis
# Maps variants to canonical forms
# =============================================================================

SYNONYMS = {
    # Machine Learning / AI
    'machine learning': 'machine_learning',
    'machine-learning': 'machine_learning',
    'deep learning': 'deep_learning',
    'deep-learning': 'deep_learning',
    'neural network': 'neural_network',
    'neural-net': 'neural_network',
    'nn': 'neural_network',

    # NLP
    'natural language processing': 'nlp',
    'natural-language-processing': 'nlp',
    'language model': 'language_model',
    'language_model': 'language_model',
    'large language model': 'llm',
    'large-language-model': 'llm',
    'llms': 'llm',

    # Computer Vision
    'computer vision': 'computer_vision',
    'computer-vision': 'computer_vision',
    'cv': 'computer_vision',
    'convolutional neural network': 'cnn',
    'convolutional neural net': 'cnn',
    'cnn': 'cnn',
    'vision transformer': 'vit',
    'vit': 'vit',

    # Reinforcement Learning
    'reinforcement learning': 'rl',
    'reinforcement-learning': 'rl',
    'rl': 'rl',
    'deep reinforcement learning': 'drl',
    'drl': 'drl',

    # Model types
    'generative adversarial network': 'gan',
    'gans': 'gan',
    'vae': 'vae',
    'variational autoencoder': 'vae',
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

    # Evaluation
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
    'sota': 'sota',
    'baseline': 'baseline',
    'baselines': 'baseline',

    # Graph / Network
    'graph neural network': 'gnn',
    'gnn': 'gnn',
    'gnns': 'gnn',
    'graph convolution': 'gc',
    'gcn': 'gcn',

    # Optimization
    'stochastic gradient descent': 'sgd',
    'sgd': 'sgd',
    'adam': 'adam',
    'adamw': 'adamw',

    # Attention
    'self-attention': 'self_attention',
    'selfattention': 'self_attention',
    'cross-attention': 'cross_attention',
    'cross attention': 'cross_attention',

    # Agent / RL
    'multi-agent': 'multi_agent',
    'multi agent': 'multi_agent',
    'multiagent': 'multi_agent',
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
}

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
