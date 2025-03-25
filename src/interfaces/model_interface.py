import os
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import esm
from openfold.utils.import_weights import import_jax_weights
from openfold.model import model
from openfold.data import data_transforms
from openfold.utils.tensor_utils import tensor_tree_map
from Bio import PDB
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import protein_letters_3to1
import alphafold
from alphafold.model import model as alphafold_model
from alphafold.data import pipeline
from alphafold.data.templates import TemplateHit
import rosettafold
from rosettafold.model import RoseTTAFold
from rosettafold.utils import prepare_input
from src.core.ensemble import EnsemblePredictor, EnsembleConfig, EnsembleMethod
from src.core.model_versioning import ModelVersionManager
from src.core.memory_management import GPUMemoryManager, ModelWeightCache
import json
import time
import gc
import psutil
from src.exceptions import MemoryError, ResourceError

from src.config import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OPENFOLD_DIR,
    DEFAULT_ESM_DIR,
    SUPPORTED_FORMATS,
    ERROR_MESSAGES
)

from src.interfaces.base_interface import BaseModelInterface, BaseModelConfig
from src.interfaces.openfold_interface import OpenFoldInterface
from src.interfaces.esm_interface import ESMInterface
from src.interfaces.alphafold_interface import AlphaFoldInterface
from src.interfaces.rosettafold_interface import RoseTTAFoldInterface

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model_type: str  # 'openfold' or 'esm'
    output_format: str  # 'pdb' or 'mmcif'
    embedding_config: Dict[str, Dict[str, Any]]  # Maps layer names to their configurations
    model_path: Optional[str] = None  # Path to model weights (for OpenFold)
    config_path: Optional[str] = None  # Path to config file (for OpenFold)
    model_name: Optional[str] = None  # Name of the model (for ESM)
    hyperparameters: Optional[Dict[str, Any]] = None  # Hyperparameters for the model

class ModelInterface:
    """Main interface for protein structure prediction models."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_memory_usage: float = 0.9
    ):
        """
        Initialize model interface.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'openfold', 'esm', 'alphafold', 'rosettafold')
            model_path: Path to model weights
            config_path: Path to model configuration
            device: Device to run model on
            max_memory_usage: Maximum fraction of GPU memory to use
        """
        try:
            logger.info(f"Initializing {model_type} model interface")
            
            # Create model config
            config = BaseModelConfig(
                model_name=model_name,
                model_type=model_type,
                model_path=model_path,
                config_path=config_path,
            )
            
            # Initialize appropriate model interface
            if model_type.lower() == 'openfold':
                self.interface = OpenFoldInterface(config)
            elif model_type.lower() == 'esm':
                self.interface = ESMInterface(config)
            elif model_type.lower() == 'alphafold':
                self.interface = AlphaFoldInterface(config)
            elif model_type.lower() == 'rosettafold':
                self.interface = RoseTTAFoldInterface(config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.debug(f"Model interface initialized for {model_type}")
            
            self.device = device
            self.max_memory_usage = max_memory_usage
            self.model = None
            self.model_path = Path(model_path) if model_path else None
            
        except Exception as e:
            logger.error(f"Failed to initialize model interface: {str(e)}")
            raise

    def extract_sequence_from_pdb(
        self,
        pdb_file: str,
        chain_id: Optional[str] = None
    ) -> Union[str, Dict[str, str]]:
        """
        Extract amino acid sequence from PDB file.
        
        Args:
            pdb_file: Path to PDB file
            chain_id: Optional chain identifier. If None, returns sequences for all chains
            
        Returns:
            If chain_id is provided: amino acid sequence string for that chain
            If chain_id is None: dictionary mapping chain IDs to their sequences
        """
        try:
            logger.info(f"Extracting sequence from PDB file: {pdb_file}")
            
            # Parse PDB file
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('input', pdb_file)
            
            # Get first model
            model = structure[0]
            
            sequences = {}
            for chain in model:
                # Skip if specific chain requested and this isn't it
                if chain_id and chain.id != chain_id:
                    continue
                    
                # Extract sequence
                sequence = ""
                for residue in chain:
                    if residue.id[0] == ' ':  # Skip heteroatoms
                        try:
                            aa = protein_letters_3to1[residue.get_resname()]
                            sequence += aa
                        except KeyError:
                            logger.warning(f"Unknown residue type {residue.get_resname()} in chain {chain.id}")
                            sequence += 'X'
                
                sequences[chain.id] = sequence
                logger.debug(f"Extracted sequence of length {len(sequence)} from chain {chain.id}")
            
            if not sequences:
                raise ValueError("No valid amino acid sequences found in PDB file")
            
            # Return single sequence if chain_id specified, otherwise return all sequences
            if chain_id:
                if chain_id not in sequences:
                    raise ValueError(f"Chain {chain_id} not found in PDB file")
                return sequences[chain_id]
            return sequences
            
        except Exception as e:
            logger.error(f"Failed to extract sequence from PDB file: {str(e)}")
            raise
    
    def process_pdb_file(
        self,
        input_file: str,
        output_file: str,
        **kwargs
    ) -> None:
        """
        Process a PDB file directly without model generation.
        
        Args:
            input_file: Path to input PDB file
            output_file: Path to save the processed structure
            **kwargs: Additional arguments for processing
        """
        try:
            logger.info(f"Processing PDB file: {input_file}")
            
            # Parse input structure
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('input', input_file)
            
            # Save structure to output file
            io = PDB.PDBIO()
            io.set_structure(structure)
            io.save(output_file)
            
            logger.debug(f"Processed structure saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process PDB file: {str(e)}")
            raise
    
    def generate_structure(
        self,
        sequence: Optional[str] = None,
        output_file: Optional[str] = None,
        call_model: bool = True,
        input_pdb: Optional[str] = None,
        file_seq: bool = False,
        chain_id: Optional[str] = None,
        compare_structures: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate or process protein structure.
        
        Args:
            sequence: Input amino acid sequence (ignored if input_pdb is provided and file_seq=True)
            output_file: Path to save the output structure
            call_model: Whether to call the model for structure prediction (default: True)
            input_pdb: Optional path to input PDB file (bypasses model call if provided)
            file_seq: Whether to extract sequence from input PDB file (default: False)
            chain_id: Chain ID to use when extracting sequence from PDB file
            compare_structures: Whether to perform structure comparison when using file_seq
            **kwargs: Additional arguments for structure generation
            
        Returns:
            If compare_structures=True and file_seq=True: Dictionary containing comparison metrics
            Otherwise: None
        """
        try:
            comparison_results = None
            
            # Handle file_seq functionality
            if file_seq and input_pdb:
                logger.info("Extracting sequence from input PDB file")
                sequence = self.extract_sequence_from_pdb(input_pdb, chain_id)
                if isinstance(sequence, dict):
                    if len(sequence) == 1:
                        sequence = next(iter(sequence.values()))
                    else:
                        raise ValueError("Multiple chains found in PDB file. Please specify chain_id.")
                
                # Generate output filename if not provided
                if not output_file:
                    output_dir = os.path.dirname(input_pdb)
                    base_name = os.path.splitext(os.path.basename(input_pdb))[0]
                    output_file = os.path.join(output_dir, f"{base_name}_predicted.pdb")
                
                # Generate structure using extracted sequence
                if call_model:
                    logger.info(f"Generating structure for extracted sequence of length {len(sequence)}")
                    self.interface.generate_structure(sequence, output_file, **kwargs)
                    
                    # Perform structure comparison if requested
                    if compare_structures:
                        from ..utils.structure_comparison import compare_structures
                        comparison_results = compare_structures(input_pdb, output_file)
                        logger.info("Structure comparison completed")
                
            # Handle normal structure generation or PDB processing
            elif input_pdb and not call_model:
                logger.info("Bypassing model call and processing PDB file directly")
                self.process_pdb_file(input_pdb, output_file, **kwargs)
            elif call_model and sequence:
                logger.info(f"Generating structure for sequence of length {len(sequence)}")
                self.interface.generate_structure(sequence, output_file, **kwargs)
                logger.debug(f"Structure saved to: {output_file}")
            else:
                raise ValueError("Either call_model must be True with a sequence, or input_pdb must be provided")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to generate/process structure: {str(e)}")
            raise
    
    def extract_embeddings(self, sequence: str) -> Dict[str, Any]:
        """
        Extract embeddings from model.
        
        Args:
            sequence: Input amino acid sequence
            
        Returns:
            Dictionary containing model embeddings
        """
        try:
            logger.info("Extracting embeddings")
            embeddings = self.interface.extract_embeddings(sequence)
            logger.debug("Successfully extracted embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {str(e)}")
            raise
    
    def optimize_memory_usage(self) -> None:
        """Optimize GPU memory usage."""
        if self.device == 'cuda':
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            if memory_allocated / memory_reserved > self.max_memory_usage:
                raise MemoryError(
                    f"GPU memory usage ({memory_allocated/1e9:.2f}GB) exceeds "
                    f"maximum allowed ({self.max_memory_usage*100}%)"
                )
                
            logger.info(
                f"GPU memory: allocated={memory_allocated/1e9:.2f}GB, "
                f"reserved={memory_reserved/1e9:.2f}GB"
            )
            
    def quantize_model(self, quantization_dtype: torch.dtype = torch.float16) -> None:
        """
        Quantize model to reduce memory usage.
        
        Args:
            quantization_dtype: Target dtype for quantization
        """
        if self.model is None:
            raise ResourceError("Model not loaded")
            
        if self.device == 'cuda':
            self.model.to(quantization_dtype)
            logger.info(f"Model quantized to {quantization_dtype}")
            
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            if self.device == 'cuda':
                self.model.cpu()
                torch.cuda.empty_cache()
            del self.model
            self.model = None
            
        gc.collect()
        
        if self.device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated()
            logger.info(f"Cleaned up resources. GPU memory allocated: {memory_allocated/1e9:.2f}GB")
            
    def validate_api_response(self):
        """Validate API responses from external models"""
        pass
        
    def handle_api_rate_limits(self):
        """Handle API rate limits and quotas"""
        pass

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()