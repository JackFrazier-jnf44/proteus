"""Model quantization module for reducing memory usage."""

from .quantization_manager import QuantizationManager, QuantizationConfig

__all__ = [
    'QuantizationManager',
    'QuantizationConfig'
]