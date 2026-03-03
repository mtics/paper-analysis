"""
Unified logging system for CCF analysis.
Logs are saved to output/logs directory with rotation.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


# Default log directory
DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "output" / "logs"


def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_dir: Path = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.

    Args:
        name: Logger name (defaults to root logger)
        level: Logging level
        log_dir: Directory for log files (defaults to output/logs)
        console: Whether to add console handler

    Returns:
        Configured logger instance
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"ccf_analysis_{timestamp}.log"

    # File handler with rotation (10MB max, 5 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    Uses hierarchical naming: analysis.features.trends -> "trends"
    """
    # Simplify logger names
    if name.startswith('analysis.'):
        parts = name.split('.')
        # Get the meaningful part
        if 'features' in parts:
            idx = parts.index('features')
            if idx + 1 < len(parts):
                name = parts[idx + 1]
            else:
                name = 'analysis'
        else:
            name = parts[-1]

    return logging.getLogger(name)


# Default logger for quick access
default_logger = setup_logger('analysis')
