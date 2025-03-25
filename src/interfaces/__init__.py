"""Model interfaces for different protein structure prediction models."""

from src.interfaces.base_interface import BaseModelInterface, BaseModelConfig
from src.interfaces.model_interface import ModelInterface, ModelConfig
from src.interfaces.alphafold_interface import AlphaFoldInterface
from src.interfaces.esm_interface import ESMInterface
from src.interfaces.openfold_interface import OpenFoldInterface
from src.interfaces.rosettafold_interface import RoseTTAFoldInterface
from src.interfaces.colabfold_interface import ColabFoldInterface

__all__ = [
    # Base interfaces
    'BaseModelInterface',
    'BaseModelConfig',
    'ModelInterface',
    'ModelConfig',
    
    # Model-specific interfaces
    'AlphaFoldInterface',
    'ESMInterface',
    'OpenFoldInterface',
    'RoseTTAFoldInterface',
    'ColabFoldInterface'
]