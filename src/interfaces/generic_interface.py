"""Generic interface for similar model implementations."""

import torch
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO, Structure

from .base_interface import (
    BaseModelInterface,
    BaseModelConfig,
    ModelError,
    ModelInitializationError,
    ModelInferenceError,
    ModelMemoryError
)
from ..exceptions import TemplateError

logger = logging.getLogger(__name__)

class GenericModelInterface(BaseModelInterface):
    """Generic interface for similar model implementations."""
    
    def __init__(
        self,
        config: BaseModelConfig,
        model_module: Any,
        data_pipeline: Any,
        atom_types: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize generic model interface.
        
        Args:
            config: Model configuration
            model_module: Module containing the model class
            data_pipeline: Module containing the data pipeline
            atom_types: List of atom types to use
            **kwargs: Additional arguments
        """
        try:
            self.model_module = model_module
            self.data_pipeline = data_pipeline
            self.atom_types = atom_types or ['N', 'CA', 'C', 'O', 'CB']
            super().__init__(config)
            logger.info(f"Successfully initialized generic model interface for {config.model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize generic model interface: {str(e)}")
            raise ModelInitializationError(f"Generic model interface initialization failed: {str(e)}")

    def _initialize_model(self, **kwargs) -> Any:
        """Initialize the model using the provided model module."""
        try:
            logger.info(f"Initializing {self.config.model_type} model")
            
            # Get model class from module
            model_class = getattr(self.model_module, self.config.model_type.capitalize())
            
            # Initialize model with config
            model_instance = model_class(str(self.config.config_path))
            
            # Load weights if available
            if self.config.model_path:
                model_instance.load_state_dict(torch.load(self.config.model_path))
            
            # Set model to evaluation mode
            model_instance.eval()
            
            # Initialize data pipeline if available
            if hasattr(self.data_pipeline, 'DataPipeline'):
                self.data_pipeline = self.data_pipeline.DataPipeline(
                    template_featurizer=None,  # Will be initialized when needed
                    msa_featurizer=None,  # Will be initialized when needed
                )
            
            logger.debug(f"{self.config.model_type} model initialized successfully")
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.model_type} model: {str(e)}")
            raise ModelInitializationError(f"Model initialization failed: {str(e)}")

    def generate_structure(
        self,
        sequence: str,
        output_file: str,
        use_templates: Optional[bool] = None,
        template_pdbs: Optional[List[str]] = None,
        template_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Generate protein structure prediction.
        
        Args:
            sequence: Input amino acid sequence
            output_file: Path to save the output structure
            use_templates: Whether to use template information
            template_pdbs: List of template PDB files
            template_sequences: List of template sequences
            **kwargs: Additional arguments for structure generation
            
        Raises:
            TemplateError: If template processing fails
            ModelInferenceError: If structure generation fails
        """
        try:
            logger.info(f"Generating structure for sequence of length {len(sequence)}")
            
            # Use provided template setting or config default
            use_templates = use_templates if use_templates is not None else self.config.use_templates
            
            # Process templates if requested
            template_features = {}
            if use_templates:
                try:
                    template_features = self._process_templates(
                        sequence,
                        template_pdbs,
                        template_sequences
                    )
                except Exception as e:
                    logger.error(f"Template processing failed: {str(e)}")
                    raise TemplateError(f"Failed to process templates: {str(e)}")
            
            # Prepare input data
            input_data = self._prepare_input(
                sequence,
                template_features=template_features if use_templates else None,
                **kwargs
            )
            
            # Move data to device
            input_data = self.move_to_device(input_data)
            
            # Generate structure
            with torch.no_grad():
                outputs = self.model(**input_data)
            
            # Process outputs
            structure = self._process_outputs(outputs, sequence)
            
            # Save structure
            self.save_structure(structure, output_file)
            
            # Save confidence scores if available
            if hasattr(outputs, 'confidence_scores'):
                confidence_file = str(Path(output_file).with_suffix('.json'))
                self.save_confidence_scores(outputs.confidence_scores, confidence_file)
            
            logger.info(f"Structure successfully generated and saved to {output_file}")
            
        except TemplateError as e:
            logger.error(f"Template processing failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate structure: {str(e)}")
            raise ModelInferenceError(f"Structure generation failed: {str(e)}")

    def extract_embeddings(
        self,
        sequence: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract embeddings from the model."""
        try:
            logger.info("Extracting embeddings")
            
            # Prepare input data
            input_data = self._prepare_input(sequence, **kwargs)
            
            # Move data to device
            input_data = self.move_to_device(input_data)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**input_data)
            
            # Process embeddings
            embeddings = self._process_embeddings(outputs)
            
            logger.debug("Successfully extracted embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {str(e)}")
            raise ModelInferenceError(f"Embedding extraction failed: {str(e)}")

    def _prepare_input(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Prepare input data for the model.
        
        Args:
            sequence: Input amino acid sequence
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model inputs
        """
        try:
            # Get base sequence inputs
            inputs = super()._prepare_sequence_input(sequence, **kwargs)
            
            # Add model-specific preprocessing
            processed_inputs = self.data_pipeline.preprocess(
                sequence=sequence,
                **inputs,
                **kwargs
            )
            
            # Handle template information if provided
            if kwargs.get('use_templates'):
                template_data = self._process_templates(
                    sequence,
                    template_pdbs=kwargs.get('template_pdbs'),
                    template_sequences=kwargs.get('template_sequences')
                )
                processed_inputs.update(template_data)
                
            return processed_inputs
            
        except Exception as e:
            logger.error(f"Failed to prepare input data: {str(e)}")
            raise ModelInferenceError(f"Input preparation failed: {str(e)}")
    
    def _process_outputs(self, outputs: Any, sequence: str) -> Structure:
        """Process model outputs into a structure.
        
        Args:
            outputs: Raw model outputs
            sequence: Input amino acid sequence
            
        Returns:
            Bio.PDB Structure object
        """
        try:
            # Process outputs through data pipeline
            processed_outputs = self.data_pipeline.postprocess(
                outputs=outputs,
                sequence=sequence
            )
            
            # Extract atomic coordinates
            if 'atom_positions' not in processed_outputs:
                raise ValueError("No atomic coordinates found in processed outputs")
                
            coords = processed_outputs['atom_positions']
            
            # Create structure
            structure = self.decoder.create_structure(
                coords=coords,
                sequence=sequence,
                atom_types=self.atom_types
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to process model outputs: {str(e)}")
            raise ModelInferenceError(f"Output processing failed: {str(e)}")
    
    def _process_embeddings(self, outputs: Any) -> Dict[str, Any]:
        """Process model outputs into embeddings.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Dictionary containing embeddings
        """
        try:
            # Process outputs through data pipeline
            processed_outputs = self.data_pipeline.postprocess_embeddings(outputs)
            
            # Extract embeddings using decoder
            embeddings = self.decoder.decode_embeddings(processed_outputs)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to process embeddings: {str(e)}")
            raise ModelInferenceError(f"Embedding processing failed: {str(e)}")
    
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
            Dictionary containing processed template data
        """
        try:
            if not template_pdbs or not template_sequences:
                return {}
                
            if len(template_pdbs) != len(template_sequences):
                raise TemplateError(
                    f"Number of template PDBs ({len(template_pdbs)}) does not match "
                    f"number of sequences ({len(template_sequences)})"
                )
                
            # Process each template
            template_data = []
            for pdb_file, temp_seq in zip(template_pdbs, template_sequences):
                # Parse template structure
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure('template', pdb_file)
                
                # Extract coordinates
                coords = []
                for residue in structure.get_residues():
                    if residue.id[0] == ' ':  # Skip heteroatoms
                        residue_coords = []
                        for atom_type in self.atom_types:
                            if atom_type in residue:
                                residue_coords.append(residue[atom_type].get_coord())
                            else:
                                residue_coords.append(np.zeros(3))
                        coords.append(residue_coords)
                
                template_data.append({
                    'coordinates': np.array(coords),
                    'sequence': temp_seq
                })
                
            # Process templates through data pipeline
            processed_templates = self.data_pipeline.process_templates(
                target_sequence=sequence,
                template_data=template_data
            )
            
            return processed_templates
            
        except Exception as e:
            logger.error(f"Failed to process templates: {str(e)}")
            raise TemplateError(f"Template processing failed: {str(e)}") 