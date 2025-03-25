"""Distributed computing utilities for Proteus."""

from .distributed_inference import (
    DistributedInferenceManager,
    DistributedConfig,
    DistributionStrategy
)

__all__ = [
    'DistributedInferenceManager',
    'DistributedConfig',
    'DistributionStrategy'
] 