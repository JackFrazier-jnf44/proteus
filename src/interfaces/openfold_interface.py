"""Interface for OpenFold model."""

import torch
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np

from openfold.model import model
from openfold.data import data_transforms
from openfold.utils.tensor_utils import tensor_tree_map
from Bio import PDB
from Bio.PDB import PDBIO, Structure

from .generic_interface import GenericModelInterface
from .base_interface import BaseModelConfig

logger = logging.getLogger(__name__)

class OpenFoldInterface(GenericModelInterface):
    """Interface for OpenFold model."""
    
    def __init__(self, config: BaseModelConfig, **kwargs):
        """Initialize OpenFold interface."""
        super().__init__(
            config=config,
            model_module=model,
            data_pipeline=data_transforms,
            atom_types=['N', 'CA', 'C', 'O', 'CB'],  # OpenFold uses these 5 atoms
            **kwargs
        )
    
    def _initialize_model(self, **kwargs) -> Any:
        """
        Initialize OpenFold model.
        
        Args:
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Initialized OpenFold model instance
        """
        try:
            logger.info("Initializing OpenFold model")
            
            # Load OpenFold model
            model_instance = model.AlphaFold(str(self.config_path))
            model_instance.load_state_dict(torch.load(self.model_path))
            model_instance.eval()
            
            # Initialize data pipeline
            self.data_pipeline = data_transforms.DataPipeline(
                template_featurizer=None,  # Will be initialized when needed
                msa_featurizer=None,  # Will be initialized when needed
            )
            
            logger.debug(f"OpenFold model initialized")
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenFold model: {str(e)}")
            raise
    
    def generate_structure(
        self,
        sequence: str,
        output_file: str,
        num_recycles: int = 3,
        use_templates: bool = False,
        **kwargs
    ) -> None:
        """
        Generate protein structure prediction.
        
        Args:
            sequence: Input amino acid sequence
            output_file: Path to save the output structure
            num_recycles: Number of recycling iterations
            use_templates: Whether to use template information
            **kwargs: Additional arguments for structure generation
        """
        try:
            logger.info(f"Generating structure for sequence of length {len(sequence)}")
            
            # Prepare features
            features = self.data_pipeline.process(
                sequence=sequence,
                description="query",
                use_templates=use_templates
            )
            
            # Move features to device
            features = self.move_to_device(features)
            
            # Generate structure
            with torch.no_grad():
                outputs = self.model(
                    features,
                    is_training=False,
                    compute_loss=False,
                    num_recycle=num_recycles
                )
                
            # Convert output to structure
            final_atom_positions = outputs['structure_module']['final_atom_positions']
            atom_types = ['N', 'CA', 'C', 'O', 'CB']  # OpenFold uses these 5 atoms
            structure = self.create_structure(
                final_atom_positions.cpu().numpy(),
                sequence,
                atom_types
            )
            
            # Save structure and confidence scores
            self.save_structure(structure, output_file)
            self.save_confidence_scores(outputs, output_file.replace('.pdb', '_confidence.json'))
            
            logger.debug(f"Structure saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate structure: {str(e)}")
            raise
    
    def extract_embeddings(
        self,
        sequence: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract embeddings from the model.
        
        Args:
            sequence: Input amino acid sequence
            **kwargs: Additional arguments for embedding extraction
            
        Returns:
            Dictionary containing embeddings and attention maps
        """
        try:
            logger.info(f"Extracting embeddings for sequence of length {len(sequence)}")
            
            # Prepare features
            features = self.data_pipeline.process(
                sequence=sequence,
                description="query",
                use_templates=False
            )
            
            # Move features to device
            features = self.move_to_device(features)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(
                    features,
                    is_training=False,
                    compute_loss=False,
                    return_embeddings=True
                )
                
            # Process embeddings
            embeddings = {
                'sequence': sequence,
                'msa_embedding': outputs['msa_embedding'].cpu().numpy(),
                'pair_embedding': outputs['pair_embedding'].cpu().numpy(),
                'single_embedding': outputs['single_embedding'].cpu().numpy(),
                'attention_maps': outputs.get('attention_maps', {}).cpu().numpy()
            }
            
            logger.debug("Successfully extracted embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {str(e)}")
            raise