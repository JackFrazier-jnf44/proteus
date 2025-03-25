"""Base decoder for model outputs."""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from Bio.PDB import Structure
import torch
import logging

from src.exceptions import DecoderError

logger = logging.getLogger(__name__)

class BaseDecoder:
    """Base class for decoding model outputs."""
    
    def __init__(self, model_type: str):
        """Initialize base decoder.
        
        Args:
            model_type: Type of model this decoder handles
        """
        self.model_type = model_type
        self.atom_types = ['N', 'CA', 'C', 'O', 'CB']
        self.min_sequence_length = 1
        self.max_sequence_length = 2500  # Default max length
    
    def validate_sequence_length(self, sequence: str) -> bool:
        """Validate input sequence length.
        
        Args:
            sequence: Input amino acid sequence
            
        Returns:
            True if valid, raises DecoderError otherwise
        """
        try:
            seq_len = len(sequence)
            if seq_len < self.min_sequence_length:
                raise DecoderError(
                    f"Sequence length ({seq_len}) is below minimum allowed "
                    f"length ({self.min_sequence_length})"
                )
            
            if seq_len > self.max_sequence_length:
                raise DecoderError(
                    f"Sequence length ({seq_len}) exceeds maximum allowed "
                    f"length ({self.max_sequence_length})"
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Sequence validation failed: {str(e)}")
            raise DecoderError(f"Sequence validation failed: {str(e)}")
    
    def set_sequence_length_limits(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> None:
        """Set sequence length limits.
        
        Args:
            min_length: Minimum allowed sequence length
            max_length: Maximum allowed sequence length
        """
        if min_length is not None:
            if min_length < 1:
                raise ValueError("Minimum sequence length must be at least 1")
            self.min_sequence_length = min_length
            
        if max_length is not None:
            if max_length < self.min_sequence_length:
                raise ValueError(
                    f"Maximum sequence length ({max_length}) must be greater "
                    f"than minimum length ({self.min_sequence_length})"
                )
            self.max_sequence_length = max_length
    
    def decode_structure(
        self,
        outputs: Dict[str, Any],
        sequence: str,
        **kwargs
    ) -> Tuple[Structure, np.ndarray, np.ndarray]:
        """Decode model outputs into structure and confidence metrics.
        
        Args:
            outputs: Raw model outputs
            sequence: Input amino acid sequence
            **kwargs: Additional arguments for decoding
            
        Returns:
            Tuple containing:
            - Predicted structure
            - Confidence scores
            - Distance matrix
        """
        # Validate sequence length first
        self.validate_sequence_length(sequence)
        
        # Extract atomic coordinates and confidence scores
        coords = outputs.get('atom_positions')
        confidence = outputs.get('confidence_scores', np.zeros(len(sequence)))
        
        # Calculate distance matrix if not provided
        dist_matrix = outputs.get('distance_matrix')
        if dist_matrix is None and coords is not None:
            dist_matrix = self._calculate_distance_matrix(coords)
        
        # Create Bio.PDB structure
        if coords is not None:
            structure = self.create_structure(coords, sequence)
        else:
            raise ValueError("No atomic coordinates found in model outputs")
            
        return structure, confidence, dist_matrix
    
    def decode_embeddings(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode model outputs into embeddings.
        
        Args:
            outputs: Raw model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing embeddings
        """
        embeddings = {}
        
        # Extract per-residue embeddings
        if 'residue_embeddings' in outputs:
            embeddings['residue'] = outputs['residue_embeddings']
            
        # Extract sequence-level embeddings
        if 'sequence_embedding' in outputs:
            embeddings['sequence'] = outputs['sequence_embedding']
            
        # Extract attention maps if available
        if 'attention_maps' in outputs:
            embeddings['attention'] = outputs['attention_maps']
            
        # Extract other embedding types based on model output
        for key, value in outputs.items():
            if 'embedding' in key.lower() and key not in ['residue_embeddings', 'sequence_embedding']:
                embeddings[key] = value
                
        if not embeddings:
            raise ValueError("No embeddings found in model outputs")
            
        return embeddings
    
    def decode_confidence(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode confidence scores from model outputs.
        
        Args:
            outputs: Raw model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing confidence metrics
        """
        confidence_metrics = {}
        
        # Extract pLDDT scores if available
        if 'plddt' in outputs:
            confidence_metrics['plddt'] = outputs['plddt']
            
        # Extract PAE scores if available
        if 'pae' in outputs:
            confidence_metrics['pae'] = outputs['pae']
            
        # Extract per-residue confidence scores
        if 'confidence_scores' in outputs:
            confidence_metrics['per_residue'] = outputs['confidence_scores']
            
        # Calculate overall model confidence
        if 'plddt' in outputs:
            confidence_metrics['global_confidence'] = np.mean(outputs['plddt'])
        elif 'confidence_scores' in outputs:
            confidence_metrics['global_confidence'] = np.mean(outputs['confidence_scores'])
            
        # Add any additional confidence metrics
        for key, value in outputs.items():
            if 'confidence' in key.lower() and key not in ['confidence_scores']:
                confidence_metrics[key] = value
                
        if not confidence_metrics:
            raise ValueError("No confidence metrics found in model outputs")
            
        return confidence_metrics
    
    def _calculate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Calculate distance matrix from atomic coordinates.
        
        Args:
            coords: Array of atomic coordinates with shape (L, A, 3)
            
        Returns:
            Distance matrix with shape (L, L)
        """
        # Use CA atoms for distance calculation
        ca_idx = self.atom_types.index('CA')
        ca_coords = coords[:, ca_idx, :]
        
        # Calculate pairwise distances
        diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
        
        return dist_matrix
    
    def create_structure(
        self,
        coords: np.ndarray,
        sequence: str,
        atom_types: Optional[List[str]] = None
    ) -> Structure:
        """Create Bio.PDB Structure from coordinates.
        
        Args:
            coords: Array of atomic coordinates with shape (L, A, 3) where:
                   L is sequence length
                   A is number of atoms per residue
                   3 is for x,y,z coordinates
            sequence: Amino acid sequence
            atom_types: Optional list of atom types to use (defaults to self.atom_types)
            
        Returns:
            Bio.PDB Structure object
        """
        try:
            from Bio.PDB import Structure, Model, Chain, Residue, Atom
            from Bio.PDB.vectors import Vector
            from Bio.Data.IUPACData import protein_letters_3to1
            
            # Validate inputs
            if len(sequence) != coords.shape[0]:
                raise ValueError(f"Sequence length ({len(sequence)}) does not match coordinates shape ({coords.shape[0]})")
            
            # Use provided atom types or default
            atom_types = atom_types or self.atom_types
            if coords.shape[1] != len(atom_types):
                raise ValueError(f"Number of atoms ({coords.shape[1]}) does not match atom types ({len(atom_types)})")
            
            # Create structure hierarchy
            structure = Structure('S')
            model = Model(0)
            chain = Chain('A')
            
            # Add residues
            for i, (aa, residue_coords) in enumerate(zip(sequence, coords)):
                # Create residue
                res_id = (' ', i, ' ')  # Standard PDB residue ID format
                residue = Residue(res_id, aa, ' ')
                
                # Add atoms
                for j, (atom_type, coord) in enumerate(zip(atom_types, residue_coords)):
                    # Create atom
                    atom = Atom(
                        atom_type,
                        coord,
                        20.0,  # Default B-factor
                        1.0,   # Default occupancy
                        ' ',   # Default altloc
                        atom_type,
                        i,     # Element number
                        'C'    # Default element
                    )
                    residue.add(atom)
                
                chain.add(residue)
            
            model.add(chain)
            structure.add(model)
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to create structure: {str(e)}")
            raise DecoderError(f"Failed to create structure: {str(e)}")
    
    def to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        """Convert tensor to numpy array.
        
        Args:
            tensor: Input tensor (torch.Tensor, numpy.ndarray, or None)
            
        Returns:
            Numpy array or None if input is None
        """
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        raise DecoderError(f"Unsupported tensor type: {type(tensor)}")

    def validate_outputs(
        self,
        outputs: Dict[str, Any],
        required_keys: Optional[List[str]] = None
    ) -> bool:
        """Validate model outputs.
        
        Args:
            outputs: Raw model outputs
            required_keys: Optional list of required keys
            
        Returns:
            True if valid, raises DecoderError otherwise
        """
        try:
            if not isinstance(outputs, dict):
                raise DecoderError(f"Expected dict outputs, got {type(outputs)}")
            
            if required_keys:
                missing = [k for k in required_keys if k not in outputs]
                if missing:
                    raise DecoderError(f"Missing required keys: {missing}")
                
            return True
        
        except Exception as e:
            logger.error(f"Output validation failed: {str(e)}")
            raise DecoderError(f"Output validation failed: {str(e)}")

    def _validate_structure_outputs(self, structure_outputs: Dict[str, Any]) -> None:
        """Validate structure module outputs.
        
        Args:
            structure_outputs: Structure module outputs to validate
        """
        required_keys = ['final_atom_positions']
        
        # Check for required keys
        missing_keys = [k for k in required_keys if k not in structure_outputs]
        if missing_keys:
            raise DecoderError(f"Missing required structure keys: {missing_keys}")
        
        # Validate coordinates shape
        coords = structure_outputs['final_atom_positions']
        if not isinstance(coords, (np.ndarray, torch.Tensor)):
            raise DecoderError(f"Expected coordinates to be array or tensor, got {type(coords)}")
        
        if len(coords.shape) != 3:
            raise DecoderError(f"Expected 3D coordinates (L,A,3), got shape {coords.shape}")

    def _validate_embedding_outputs(self, outputs: Dict[str, Any]) -> None:
        """Validate embedding outputs.
        
        Args:
            outputs: Embedding outputs to validate
        """
        # Check for at least one type of embedding
        embedding_keys = ['embeddings', 'representations', 'msa_embedding', 'pair_embedding']
        if not any(k in outputs for k in embedding_keys):
            raise DecoderError("No embedding outputs found")
        
        # Validate embedding shapes
        for key in embedding_keys:
            if key in outputs:
                emb = outputs[key]
                if not isinstance(emb, (np.ndarray, torch.Tensor)):
                    raise DecoderError(f"Expected {key} to be array or tensor, got {type(emb)}")