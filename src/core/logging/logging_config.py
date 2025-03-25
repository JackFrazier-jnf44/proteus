"""Logging configuration for the Proteus framework."""

import logging
import logging.handlers
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = 'INFO',
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        console_output: Whether to output logs to console
    """
    # Create log directory if specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create formatters
    structured_formatter = StructuredFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log directory is specified
    if log_dir:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / 'multi_model_analysis.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(structured_formatter)
        root_logger.addHandler(file_handler)
        
        # Add error file handler for ERROR and above
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / 'error.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(structured_formatter)
        root_logger.addHandler(error_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a message with additional context.
    
    Args:
        logger: Logger instance
        level: Logging level
        message: Log message
        extra: Additional context to include in the log
    """
    if extra is None:
        extra = {}
    
    logger.log(level, message, extra=extra) 