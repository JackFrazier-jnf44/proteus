"""RoseTTAFold model output decoder."""

from typing import Dict, Any, Tuple
import numpy as np
from Bio.PDB import Structure
import logging

from .base import BaseDecoder
from src.exceptions import DecoderError

logger = logging.getLogger(__name__)

class RoseTTAFoldDecoder(BaseDecoder):
    """Decoder for RoseTTAFold model outputs."""
    
    def __init__(self):
        """Initialize RoseTTAFold decoder."""
        super().__init__('rosettafold')
        self.atom_types = ['N', 'CA', 'C', 'O', 'CB']  # RoseTTAFold uses these 5 atoms
    
    def decode_structure(
        self,
        outputs: Dict[str, Any],
        sequence: str,
        **kwargs
    ) -> Tuple[Structure, np.ndarray, np.ndarray]:
        """Decode RoseTTAFold outputs into structure and confidence metrics.
        
        Args:
            outputs: Raw RoseTTAFold model outputs
            sequence: Input amino acid sequence
            **kwargs: Additional arguments for decoding
            
        Returns:
            Tuple containing:
            - Predicted structure
            - Confidence scores (pLDDT)
            - Distance matrix
        """
        try:
            # Extract coordinates from structure module
            final_atom_positions = outputs['structure_module']['final_atom_positions']
            coords = self.to_numpy(final_atom_positions)
            
            # Create structure
            structure = self.create_structure(coords, sequence)
            
            # Extract confidence scores
            confidence = self.to_numpy(outputs['structure_module']['plddt'])
            
            # Extract distance matrix
            distance_matrix = self.to_numpy(outputs['structure_module']['predicted_distogram'])
            
            return structure, confidence, distance_matrix
            
        except Exception as e:
            logger.error(f"Failed to decode RoseTTAFold structure: {str(e)}")
            raise DecoderError(f"Failed to decode RoseTTAFold structure: {str(e)}")
    
    def decode_embeddings(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode RoseTTAFold outputs into embeddings.
        
        Args:
            outputs: Raw RoseTTAFold model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing embeddings
        """
        try:
            embeddings = {
                'msa_embedding': self.to_numpy(outputs['msa_embedding']),
                'pair_embedding': self.to_numpy(outputs['pair_embedding']),
                'single_embedding': self.to_numpy(outputs['single_embedding']),
                'structure_embedding': self.to_numpy(outputs['structure_module']['single_embedding'])
            }
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to decode RoseTTAFold embeddings: {str(e)}")
            raise DecoderError(f"Failed to decode RoseTTAFold embeddings: {str(e)}")
    
    def decode_confidence(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode confidence scores from RoseTTAFold outputs.
        
        Args:
            outputs: Raw RoseTTAFold model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing confidence metrics
        """
        try:
            confidence = {
                'plddt': self.to_numpy(outputs['structure_module']['plddt']),
                'pae': self.to_numpy(outputs['structure_module']['predicted_aligned_error']),
                'ptm': self.to_numpy(outputs['structure_module']['ptm']),
                'model_confidence': self.to_numpy(outputs['structure_module']['model_confidence'])
            }
            
            return confidence
            
        except Exception as e:
            logger.error(f"Failed to decode RoseTTAFold confidence scores: {str(e)}")
            raise DecoderError(f"Failed to decode RoseTTAFold confidence scores: {str(e)}") 