"""Interface for ESM model."""

import torch
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np

import esm
from esm import Alphabet

from .generic_interface import GenericModelInterface
from .base_interface import BaseModelConfig

logger = logging.getLogger(__name__)

class ESMInterface(GenericModelInterface):
    """Interface for ESM model."""
    
    def __init__(self, config: BaseModelConfig, **kwargs):
        """Initialize ESM interface."""
        super().__init__(
            config=config,
            model_module=esm,
            data_pipeline=None,  # ESM doesn't use a data pipeline
            atom_types=['N', 'CA', 'C'],  # ESM predicts backbone atoms
            **kwargs
        )
        self.alphabet = None
    
    def _initialize_model(self, **kwargs) -> Any:
        """Initialize the ESM model."""
        try:
            logger.info("Initializing ESM model")
            
            # Load ESM model
            model_instance, alphabet = esm.pretrained.load_model_and_alphabet(
                self.config.model_name or "esm2_t33_650M_UR50D"
            )
            model_instance.eval()
            
            # Store alphabet for tokenization
            self.alphabet = alphabet
            
            logger.debug(f"ESM model initialized: {self.config.model_name}")
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize ESM model: {str(e)}")
            raise
    
    def generate_structure(
        self,
        sequence: str,
        output_file: str,
        **kwargs
    ) -> None:
        """Generate protein structure prediction."""
        try:
            logger.info(f"Generating structure for sequence of length {len(sequence)}")
            
            # Tokenize sequence
            batch_converter = self.alphabet.get_batch_converter()
            _, _, tokens = batch_converter([(None, sequence)])
            
            # Move to device
            tokens = self.move_to_device(tokens)
            
            # Generate structure
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[self.model.num_layers])
                
            # Extract coordinates and create structure
            coords = results["positions"][-1, :len(sequence)]  # [L, 3, 3]
            structure = self.create_structure(coords.cpu().numpy(), sequence, self.atom_types)
            
            # Save structure and confidence scores
            self.save_structure(structure, output_file)
            self.save_confidence_scores(results, output_file.replace('.pdb', '_confidence.json'))
            
            logger.debug(f"Structure saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate structure: {str(e)}")
            raise
    
    def extract_embeddings(
        self,
        sequence: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract embeddings from the model."""
        try:
            logger.info(f"Extracting embeddings for sequence of length {len(sequence)}")
            
            # Tokenize sequence
            batch_converter = self.alphabet.get_batch_converter()
            _, _, tokens = batch_converter([(None, sequence)])
            
            # Move to device
            tokens = self.move_to_device(tokens)
            
            # Extract embeddings
            with torch.no_grad():
                results = self.model(
                    tokens,
                    repr_layers=[self.model.num_layers],
                    return_contacts=True
                )
                
            # Process results
            embeddings = {
                'sequence': sequence,
                'representations': {
                    f'layer_{i}': results['representations'][i].cpu().numpy()
                    for i in results['representations']
                },
                'attentions': results.get('attentions', {}).cpu().numpy(),
                'contacts': results.get('contacts', None).cpu().numpy()
            }
            
            logger.debug("Successfully extracted embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {str(e)}")
            raise 