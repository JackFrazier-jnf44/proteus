"""ESM model output decoder."""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
from Bio.PDB import Structure
import torch
import logging
from dataclasses import dataclass

from .base import BaseDecoder
from src.exceptions import DecoderError

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_tokens_per_batch: int = 1024
    pad_to_multiple: int = 8
    
class ESMDecoder(BaseDecoder):
    """Decoder for ESM model outputs."""
    
    def __init__(self, batch_config: Optional[BatchConfig] = None):
        """Initialize ESM decoder.
        
        Args:
            batch_config: Optional batch processing configuration
        """
        super().__init__('esm')
        self.atom_types = ['N', 'CA', 'C', 'O']  # ESM only predicts backbone atoms
        self.batch_config = batch_config or BatchConfig()
    
    def decode_structure(
        self,
        outputs: Dict[str, Any],
        sequence: str,
        **kwargs
    ) -> Tuple[Structure, np.ndarray, np.ndarray]:
        """Decode ESM outputs into structure and confidence metrics.
        
        Args:
            outputs: Raw ESM model outputs
            sequence: Input amino acid sequence
            **kwargs: Additional arguments for decoding
            
        Returns:
            Tuple containing:
            - Predicted structure
            - Confidence scores (per-residue)
            - Distance matrix
        """
        try:
            # Validate outputs first
            self.validate_outputs(outputs, ['positions'])
            
            # Extract coordinates from positions
            coords = self.to_numpy(outputs['positions'][-1, :len(sequence)])
            
            # Create structure
            structure = self.create_structure(coords, sequence)
            
            # Extract confidence scores (if available)
            confidence = self.to_numpy(outputs.get('confidence', np.zeros(len(sequence))))
            
            # Extract distance matrix (if available)
            distance_matrix = self.to_numpy(outputs.get('distance_matrix', np.zeros((len(sequence), len(sequence)))))
            
            return structure, confidence, distance_matrix
            
        except Exception as e:
            logger.error(f"Failed to decode ESM structure: {str(e)}")
            raise DecoderError(f"Failed to decode ESM structure: {str(e)}")
    
    def decode_embeddings(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode ESM outputs into embeddings.
        
        Args:
            outputs: Raw ESM model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing embeddings and attention maps
        """
        try:
            # Validate outputs first
            self.validate_outputs(outputs, ['representations'])
            
            embeddings = {
                'representations': self.to_numpy(outputs['representations']),
                'attention_maps': self.to_numpy(outputs.get('attentions', None)),
                'contacts': self.to_numpy(outputs.get('contacts', None))
            }
            
            # Filter out None values
            embeddings = {k: v for k, v in embeddings.items() if v is not None}
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to decode ESM embeddings: {str(e)}")
            raise DecoderError(f"Failed to decode ESM embeddings: {str(e)}")
    
    def decode_confidence(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode confidence scores from ESM outputs.
        
        Args:
            outputs: Raw ESM model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing confidence metrics:
            - per_residue: Per-residue confidence scores
            - plddt: Predicted LDDT scores
            - attention: Attention head confidence scores
        """
        try:
            confidence = {
                'per_residue': self.to_numpy(outputs.get('confidence', None)),
                'plddt': self.to_numpy(outputs.get('plddt', None)),
                'attention': self.to_numpy(outputs.get('attention_scores', None))
            }
            
            # Filter out None values
            confidence = {k: v for k, v in confidence.items() if v is not None}
            
            if not confidence:
                logger.warning("No confidence scores found in ESM outputs")
                confidence['per_residue'] = np.zeros(len(outputs['positions'][-1]))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Failed to decode ESM confidence scores: {str(e)}")
            raise DecoderError(f"Failed to decode ESM confidence scores: {str(e)}")
    
    def decode_embeddings_batch(
        self,
        outputs_batch: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode ESM outputs into embeddings for a batch of sequences.
        
        Args:
            outputs_batch: List of raw ESM model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing batched embeddings and attention maps
        """
        try:
            # Validate batch outputs
            if not outputs_batch:
                raise DecoderError("Empty batch of outputs")
                
            # Initialize batch embeddings
            batch_embeddings = {
                'representations': [],
                'attention_maps': [],
                'contacts': []
            }
            
            # Process each output in batch
            for outputs in outputs_batch:
                # Validate outputs
                self.validate_outputs(outputs, ['representations'])
                
                # Extract embeddings
                embeddings = {
                    'representations': self.to_numpy(outputs['representations']),
                    'attention_maps': self.to_numpy(outputs.get('attentions', None)),
                    'contacts': self.to_numpy(outputs.get('contacts', None))
                }
                
                # Add to batch
                for key, value in embeddings.items():
                    if value is not None:
                        batch_embeddings[key].append(value)
            
            # Stack batch embeddings
            for key in batch_embeddings:
                if batch_embeddings[key]:
                    batch_embeddings[key] = np.stack(batch_embeddings[key])
                else:
                    del batch_embeddings[key]
            
            return batch_embeddings
            
        except Exception as e:
            logger.error(f"Failed to decode ESM batch embeddings: {str(e)}")
            raise DecoderError(f"Failed to decode ESM batch embeddings: {str(e)}")
            
    def prepare_sequence_batch(
        self,
        sequences: List[str],
        **kwargs
    ) -> List[List[str]]:
        """Prepare sequences for batch processing.
        
        Args:
            sequences: List of input sequences
            **kwargs: Additional arguments for batch preparation
            
        Returns:
            List of sequence batches
        """
        try:
            # Validate all sequences
            for seq in sequences:
                self.validate_sequence_length(seq)
                
            # Sort sequences by length for efficient batching
            sequences = sorted(sequences, key=len, reverse=True)
            
            batches = []
            current_batch = []
            current_tokens = 0
            
            for seq in sequences:
                seq_tokens = len(seq)
                
                # Check if adding sequence exceeds batch limits
                if (len(current_batch) >= self.batch_config.batch_size or
                    current_tokens + seq_tokens > self.batch_config.max_tokens_per_batch):
                    # Add current batch and start new one
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_tokens = 0
                
                # Add sequence to current batch
                current_batch.append(seq)
                current_tokens += seq_tokens
                
                # Pad to multiple if needed
                if current_tokens % self.batch_config.pad_to_multiple != 0:
                    current_tokens += (
                        self.batch_config.pad_to_multiple - 
                        (current_tokens % self.batch_config.pad_to_multiple)
                    )
            
            # Add final batch if not empty
            if current_batch:
                batches.append(current_batch)
                
            return batches
            
        except Exception as e:
            logger.error(f"Failed to prepare sequence batch: {str(e)}")
            raise DecoderError(f"Failed to prepare sequence batch: {str(e)}")
            
    def set_batch_config(self, batch_config: BatchConfig) -> None:
        """Set batch processing configuration.
        
        Args:
            batch_config: New batch configuration
        """
        self.batch_config = batch_config