import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Set up logging configuration for the package.
    
    Args:
        log_file: Path to log file. If None, logs to console only
        level: Logging level
        format: Log message format
    """
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format))
    handlers.append(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    # Set specific loggers
    loggers = [
        'multi_model_analysis',
        'multi_model_analysis.models',
        'multi_model_analysis.utils'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Prevent duplicate handlers
        logger.handlers = []
        for handler in handlers:
            logger.addHandler(handler) 