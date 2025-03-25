"""
Multi-Model Protein Structure Analysis Framework

A comprehensive framework for analyzing protein structures using multiple deep learning models,
with a focus on OpenFold and ESM. This package provides a unified interface for structure
prediction, embedding extraction, and comparative analysis.
"""

from src.main import main
from src.models.model_interface import ModelInterface, ModelConfig
from src.config import (
    DEFAULT_MODEL_CONFIGS,
    PLOT_SETTINGS,
    ANALYSIS_SETTINGS,
    SUPPORTED_FORMATS,
    ERROR_MESSAGES
)
from src.logging_config import setup_logging

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "main",
    "ModelInterface",
    "ModelConfig",
    "DEFAULT_MODEL_CONFIGS",
    "PLOT_SETTINGS",
    "ANALYSIS_SETTINGS",
    "SUPPORTED_FORMATS",
    "ERROR_MESSAGES",
    "setup_logging"
]