"""AlphaFold model output decoder."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from Bio.PDB import Structure
import logging
import hashlib
import json
from pathlib import Path
import os

from .base import BaseDecoder
from src.exceptions import DecoderError, TemplateError

logger = logging.getLogger(__name__)

class AlphaFoldDecoder(BaseDecoder):
    """Decoder for AlphaFold model outputs."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize AlphaFold decoder.
        
        Args:
            cache_dir: Optional directory for caching template features
        """
        super().__init__('alphafold')
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.proteus' / 'cache' / 'alphafold'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, template_features: Dict[str, Any]) -> str:
        """Generate cache key for template features.
        
        Args:
            template_features: Template features dictionary
            
        Returns:
            Cache key string
        """
        # Create deterministic string representation
        template_str = json.dumps(
            {k: str(v) for k, v in template_features.items()},
            sort_keys=True
        )
        return hashlib.sha256(template_str.encode()).hexdigest()
        
    def _cache_template_features(
        self,
        template_features: Dict[str, Any],
        processed_features: Dict[str, Any]
    ) -> None:
        """Cache processed template features.
        
        Args:
            template_features: Original template features
            processed_features: Processed template features
        """
        try:
            cache_key = self._get_cache_key(template_features)
            cache_file = self.cache_dir / f"{cache_key}.npz"
            
            # Save as compressed numpy array
            np.savez_compressed(
                cache_file,
                **{k: np.array(v) for k, v in processed_features.items()}
            )
            logger.debug(f"Cached template features to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache template features: {str(e)}")
            
    def _load_cached_template_features(
        self,
        template_features: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Load cached template features if available.
        
        Args:
            template_features: Template features dictionary
            
        Returns:
            Cached processed features or None if not found
        """
        try:
            cache_key = self._get_cache_key(template_features)
            cache_file = self.cache_dir / f"{cache_key}.npz"
            
            if cache_file.exists():
                with np.load(cache_file) as data:
                    cached_features = {k: data[k] for k in data.files}
                logger.debug(f"Loaded cached template features from {cache_file}")
                return cached_features
                
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load cached template features: {str(e)}")
            return None
            
    def _process_template_confidence(
        self,
        outputs: Dict[str, Any],
        template_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process template-specific confidence scores with caching.
        
        Args:
            outputs: Raw AlphaFold model outputs
            template_features: Template features dictionary
            
        Returns:
            Dictionary containing template confidence scores
            
        Raises:
            TemplateError: If template confidence processing fails
        """
        try:
            # Try to load from cache first
            cached_features = self._load_cached_template_features(template_features)
            if cached_features is not None:
                return cached_features
            
            # Process template features
            template_confidence = {}
            
            # Extract template-specific confidence scores
            if 'template_confidence' in outputs['structure_module']:
                template_confidence['template_confidence'] = self.to_numpy(
                    outputs['structure_module']['template_confidence']
                )
            
            # Extract template alignment scores
            if 'template_alignment' in outputs['structure_module']:
                template_confidence['template_alignment'] = self.to_numpy(
                    outputs['structure_module']['template_alignment']
                )
            
            # Add template feature statistics
            template_confidence['template_stats'] = {
                'num_templates': len(template_features.get('template_dgram', [])),
                'template_sequence_lengths': [
                    len(template_features.get('template_sequences', [])[i])
                    for i in range(len(template_features.get('template_sequences', [])))
                ]
            }
            
            # Cache the processed features
            self._cache_template_features(template_features, template_confidence)
            
            return template_confidence
            
        except Exception as e:
            logger.error(f"Failed to process template confidence scores: {str(e)}")
            raise TemplateError(f"Template confidence processing failed: {str(e)}")
            
    def clear_cache(self) -> None:
        """Clear the template features cache."""
        try:
            for cache_file in self.cache_dir.glob("*.npz"):
                os.remove(cache_file)
            logger.info("Cleared template features cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {str(e)}")
    
    def decode_structure(
        self,
        outputs: Dict[str, Any],
        sequence: str,
        template_features: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Structure, np.ndarray, np.ndarray]:
        """Decode AlphaFold outputs into structure and confidence metrics.
        
        Args:
            outputs: Raw AlphaFold model outputs
            sequence: Input amino acid sequence
            template_features: Optional template features dictionary
            **kwargs: Additional arguments for decoding
            
        Returns:
            Tuple containing:
            - Predicted structure
            - Confidence scores (pLDDT)
            - Distance matrix
            
        Raises:
            DecoderError: If structure decoding fails
            TemplateError: If template-related decoding fails
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
            
            # Process template information if available
            if template_features is not None:
                try:
                    # Add template-specific confidence scores
                    template_confidence = self._process_template_confidence(
                        outputs,
                        template_features
                    )
                    confidence.update(template_confidence)
                except Exception as e:
                    logger.error(f"Failed to process template confidence: {str(e)}")
                    raise TemplateError(f"Template confidence processing failed: {str(e)}")
            
            return structure, confidence, distance_matrix
            
        except TemplateError as e:
            logger.error(f"Template-related decoding failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to decode AlphaFold structure: {str(e)}")
            raise DecoderError(f"Failed to decode AlphaFold structure: {str(e)}")
    
    def decode_embeddings(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode AlphaFold outputs into embeddings.
        
        Args:
            outputs: Raw AlphaFold model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing embeddings
            
        Raises:
            DecoderError: If embedding decoding fails
        """
        try:
            embeddings = {
                'msa_embedding': self.to_numpy(outputs['msa_embedding']),
                'pair_embedding': self.to_numpy(outputs['pair_embedding']),
                'single_embedding': self.to_numpy(outputs['single_embedding']),
                'structure_embedding': self.to_numpy(outputs['structure_module']['single_embedding'])
            }
            
            # Filter out None values
            embeddings = {k: v for k, v in embeddings.items() if v is not None}
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to decode AlphaFold embeddings: {str(e)}")
            raise DecoderError(f"Failed to decode AlphaFold embeddings: {str(e)}")
    
    def decode_confidence(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Decode confidence scores from AlphaFold outputs.
        
        Args:
            outputs: Raw AlphaFold model outputs
            **kwargs: Additional arguments for decoding
            
        Returns:
            Dictionary containing confidence metrics
            
        Raises:
            DecoderError: If confidence decoding fails
        """
        try:
            confidence = {
                'plddt': self.to_numpy(outputs['structure_module']['plddt']),
                'pae': self.to_numpy(outputs['structure_module']['predicted_aligned_error']),
                'ptm': self.to_numpy(outputs['structure_module']['ptm']),
                'model_confidence': self.to_numpy(outputs['structure_module']['model_confidence'])
            }
            
            # Filter out None values
            confidence = {k: v for k, v in confidence.items() if v is not None}
            
            return confidence
            
        except Exception as e:
            logger.error(f"Failed to decode AlphaFold confidence scores: {str(e)}")
            raise DecoderError(f"Failed to decode AlphaFold confidence scores: {str(e)}") 