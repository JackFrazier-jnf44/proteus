"""Interface for AlphaFold model."""

import torch
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import numpy as np
from Bio.PDB import Structure

from alphafold.model import model
from alphafold.data import pipeline
from alphafold.common import protein
from alphafold.common import confidence

from .generic_interface import GenericModelInterface
from .base_interface import BaseModelConfig
from ..utils.decoders import AlphaFoldDecoder
from ..exceptions import TemplateError, ModelError

logger = logging.getLogger(__name__)

class AlphaFoldInterface(GenericModelInterface):
    """Interface for AlphaFold model."""
    
    def __init__(self, config: BaseModelConfig, **kwargs):
        """Initialize AlphaFold interface."""
        super().__init__(
            config=config,
            model_module=model,
            data_pipeline=pipeline,
            atom_types=protein.atom_types,
            **kwargs
        )
        self.decoder = AlphaFoldDecoder()
        self.template_featurizer = None
        self._initialize_template_featurizer()

    def _initialize_template_featurizer(self) -> None:
        """Initialize the template featurizer."""
        try:
            self.template_featurizer = pipeline.TemplateFeaturizer(
                mmcif_dir=self.config.template_mmcif_dir,
                max_template_date=self.config.max_template_date,
                max_hits=self.config.max_templates,
                kalign_binary_path=self.config.kalign_binary_path,
                release_dates_path=self.config.release_dates_path,
                obsolete_pdbs_path=self.config.obsolete_pdbs_path
            )
            logger.info("Template featurizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize template featurizer: {str(e)}")
            raise ModelError(f"Template featurizer initialization failed: {str(e)}")

    def _validate_template_inputs(
        self,
        template_pdbs: Optional[List[str]] = None,
        template_sequences: Optional[List[str]] = None
    ) -> None:
        """Validate template inputs.
        
        Args:
            template_pdbs: List of template PDB files
            template_sequences: List of template sequences
            
        Raises:
            TemplateError: If template inputs are invalid
        """
        if template_pdbs is not None:
            for pdb in template_pdbs:
                if not Path(pdb).exists():
                    raise TemplateError(f"Template PDB file not found: {pdb}")
                if not pdb.endswith('.pdb'):
                    raise TemplateError(f"Invalid template PDB file format: {pdb}")
                    
        if template_sequences is not None:
            for seq in template_sequences:
                if not all(aa in protein.standard_aa for aa in seq):
                    raise TemplateError(f"Invalid amino acid in template sequence: {seq}")

    def _process_templates(
        self,
        sequence: str,
        template_pdbs: Optional[List[str]] = None,
        template_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process template information.
        
        Args:
            sequence: Target sequence
            template_pdbs: List of template PDB files
            template_sequences: List of template sequences
            
        Returns:
            Dictionary containing processed template features
            
        Raises:
            TemplateError: If template processing fails
        """
        try:
            if template_pdbs is None and template_sequences is None:
                return {}
                
            self._validate_template_inputs(template_pdbs, template_sequences)
            
            template_features = self.template_featurizer.get_templates(
                query_sequence=sequence,
                query_pdb_code=None,
                query_release_date=None,
                hits=None,
                top_hit_only=False,
                template_pdbs=template_pdbs,
                template_sequences=template_sequences
            )
            
            return template_features
            
        except Exception as e:
            logger.error(f"Template processing failed: {str(e)}")
            raise TemplateError(f"Failed to process templates: {str(e)}")

    def _initialize_model(self, **kwargs) -> Any:
        """
        Initialize AlphaFold model.
        
        Args:
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Initialized AlphaFold model instance
        """
        try:
            logger.info("Initializing AlphaFold model")
            
            # Load AlphaFold model
            model_instance = model.AlphaFold(str(self.config_path))
            model_instance.load_state_dict(torch.load(self.model_path))
            model_instance.eval()
            
            # Initialize data pipeline with template featurizer
            self.data_pipeline = pipeline.DataPipeline(
                template_featurizer=self.template_featurizer,
                msa_featurizer=None,  # Will be initialized when needed
            )
            
            logger.debug(f"AlphaFold model initialized")
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize AlphaFold model: {str(e)}")
            raise ModelError(f"Model initialization failed: {str(e)}")
            
    def generate_structure(
        self,
        sequence: str,
        output_file: str,
        num_recycles: int = 3,
        use_templates: bool = False,
        template_pdbs: Optional[List[str]] = None,
        template_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Generate protein structure prediction.
        
        Args:
            sequence: Input amino acid sequence
            output_file: Path to save the output structure
            num_recycles: Number of recycling iterations
            use_templates: Whether to use template information
            template_pdbs: List of template PDB files
            template_sequences: List of template sequences
            **kwargs: Additional arguments for structure generation
            
        Raises:
            TemplateError: If template processing fails
            ModelError: If structure generation fails
        """
        try:
            logger.info(f"Generating structure for sequence of length {len(sequence)}")
            
            # Process templates if requested
            template_features = {}
            if use_templates:
                template_features = self._process_templates(
                    sequence,
                    template_pdbs,
                    template_sequences
                )
            
            # Prepare features
            features = self.data_pipeline.process(
                sequence=sequence,
                description="query",
                use_templates=use_templates,
                template_features=template_features
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
                
            # Decode outputs using the decoder
            structure, confidence, distance_matrix = self.decoder.decode_structure(
                outputs, 
                sequence,
                template_features=template_features if use_templates else None
            )
            
            # Save structure and confidence scores
            self.save_structure(structure, output_file)
            self.save_confidence_scores(confidence, output_file.replace('.pdb', '_confidence.json'))
            
            logger.debug(f"Structure saved to: {output_file}")
            
        except TemplateError as e:
            logger.error(f"Template processing failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate structure: {str(e)}")
            raise ModelError(f"Structure generation failed: {str(e)}")
            
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
            
        Raises:
            ModelError: If embedding extraction fails
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
                
            # Decode embeddings using the decoder
            embeddings = self.decoder.decode_embeddings(outputs)
            embeddings['sequence'] = sequence
            
            logger.debug("Successfully extracted embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {str(e)}")
            raise ModelError(f"Embedding extraction failed: {str(e)}")

    def assess_template_quality(self):
        """Assess and score template quality"""
        pass
        
    def validate_template_compatibility(self):
        """Validate template compatibility with target sequence"""
        pass