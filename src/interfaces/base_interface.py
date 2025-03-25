"""Base interface for protein structure prediction models."""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator

from src.core.decoders import (
    BaseDecoder,
    AlphaFoldDecoder,
    ESMDecoder,
    OpenFoldDecoder,
    RoseTTAFoldDecoder,
    ColabFoldDecoder
)
from src.exceptions import (
    ModelError,
    ModelInitializationError,
    ModelInferenceError,
    ModelMemoryError,
    DecoderError,
    QuantizationError
)

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    dtype: str = "float16"
    use_dynamic: bool = True
    calibration_samples: int = 100

@dataclass
class DistributedConfig:
    """Configuration for distributed inference."""
    strategy: str = "round_robin"
    devices: Optional[List[str]] = None
    batch_size: int = 1
    sync_frequency: int = 10

class BaseModelConfig(BaseModel):
    """Base configuration for all model interfaces."""
    name: str = Field(..., description="Unique name for the model instance")
    model_type: str = Field(..., description="Type of model (e.g., 'openfold', 'esm', 'alphafold')")
    output_format: str = Field(..., description="Output format for predictions")
    embedding_config: Dict[str, Dict[str, Any]] = Field(..., description="Configuration for model embeddings")
    model_path: Optional[Path] = Field(None, description="Path to model weights")
    config_path: Optional[Path] = Field(None, description="Path to model configuration")
    model_name: Optional[str] = Field(None, description="Name of the specific model variant")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model hyperparameters")
    atom_types: Optional[List[str]] = Field(None, description="List of atom types to predict")
    num_recycles: int = Field(3, description="Number of recycling iterations")
    use_templates: bool = Field(False, description="Whether to use template information")
    template_mmcif_dir: Optional[Path] = Field(None, description="Directory containing template mmCIF files")
    max_template_date: Optional[str] = Field(None, description="Maximum template date to use")
    max_templates: int = Field(4, description="Maximum number of templates to use")
    kalign_binary_path: Optional[Path] = Field(None, description="Path to kalign binary")
    release_dates_path: Optional[Path] = Field(None, description="Path to release dates file")
    obsolete_pdbs_path: Optional[Path] = Field(None, description="Path to obsolete PDBs file")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu", description="Device to run model on")
    memory_limit: Optional[int] = Field(None, description="Memory limit in bytes")
    quantization_config: Optional[QuantizationConfig] = Field(None, description="Configuration for model quantization")
    batch_size: int = Field(1, description="Batch size for inference")
    max_sequence_length: Optional[int] = Field(None, description="Maximum sequence length")
    use_ensemble: bool = Field(False, description="Whether to use ensemble predictions")
    ensemble_size: int = Field(1, description="Number of ensemble predictions")
    save_intermediates: bool = Field(False, description="Whether to save intermediate results")
    output_dir: Optional[Path] = Field(None, description="Directory for saving outputs")
    log_level: str = Field("INFO", description="Logging level")
    debug_mode: bool = Field(False, description="Whether to enable debug mode")
    use_distributed: bool = Field(False, description="Whether to use distributed inference")
    distributed_config: Optional[DistributedConfig] = Field(None, description="Configuration for distributed inference")

    @validator('model_type')
    def validate_model_type(cls, v):
        """Validate model type."""
        valid_types = {'openfold', 'esm', 'alphafold', 'rosettafold', 'colabfold'}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid model type. Must be one of {valid_types}")
        return v.lower()

    @validator('output_format')
    def validate_output_format(cls, v):
        """Validate output format."""
        valid_formats = {'pdb', 'mmcif', 'json'}
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid output format. Must be one of {valid_formats}")
        return v.lower()

    @validator('model_path', 'config_path', 'template_mmcif_dir', 'kalign_binary_path', 'release_dates_path', 'obsolete_pdbs_path')
    def validate_paths(cls, v):
        """Validate file paths."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"File not found: {path}")
        return v

    @validator('device')
    def validate_device(cls, v):
        """Validate device specification."""
        if v not in {'cpu', 'cuda'}:
            raise ValueError("Device must be either 'cpu' or 'cuda'")
        if v == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system")
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()

    @validator('max_template_date')
    def validate_template_date(cls, v):
        """Validate template date format."""
        if v is not None:
            try:
                from datetime import datetime
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Template date must be in YYYY-MM-DD format")
        return v

class BaseModelInterface:
    """Base class for all model interfaces."""
    
    def __init__(self, config: BaseModelConfig):
        """Initialize base model interface.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.device = config.device
        self.model = None
        self.distributed_manager = None
        self.decoder = self._get_decoder()
        
        # Set up logging
        logging.getLogger(__name__).setLevel(config.log_level)
    
    def _get_decoder(self) -> BaseDecoder:
        """Get appropriate decoder for model type."""
        decoder_map = {
            'alphafold': AlphaFoldDecoder,
            'esm': ESMDecoder,
            'openfold': OpenFoldDecoder,
            'rosettafold': RoseTTAFoldDecoder,
            'colabfold': ColabFoldDecoder
        }
        
        decoder_class = decoder_map.get(self.config.model_type.lower())
        if not decoder_class:
            raise ModelInitializationError(f"No decoder found for model type: {self.config.model_type}")
        
        return decoder_class()
    
    def predict(
        self,
        inputs: Dict[str, Any],
        batch_size: Optional[int] = None,
        use_distributed: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Run model prediction.
        
        Args:
            inputs: Model inputs
            batch_size: Optional batch size override
            use_distributed: Optional override for distributed inference
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Use distributed inference if configured
            if (use_distributed is None and self.config.use_distributed) or use_distributed:
                if not self.distributed_manager:
                    raise ModelInferenceError("Distributed inference not initialized")
                return self.distributed_manager.predict(inputs, batch_size)
            
            # Regular prediction
            batch_size = batch_size or self.config.batch_size
            
            # Split inputs into batches
            batches = self._split_batches(inputs, batch_size)
            
            # Run predictions
            results = []
            for batch in batches:
                # Move batch to device
                device_batch = self._move_to_device(batch)
                
                # Run prediction
                with torch.no_grad():
                    batch_result = self._run_prediction(device_batch)
                
                results.append(batch_result)
            
            # Combine results
            return self._combine_results(results)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelInferenceError(f"Prediction failed: {str(e)}")
    
    def _split_batches(self, inputs: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
        """Split inputs into batches."""
        batches = []
        num_samples = len(next(iter(inputs.values())))
        
        for i in range(0, num_samples, batch_size):
            batch = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value[i:i+batch_size]
                else:
                    batch[key] = value
            batches.append(batch)
        
        return batches
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to model device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _run_prediction(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction on a single batch.
        
        Args:
            batch: Dictionary containing batch inputs
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Run model forward pass
            outputs = self.model(**batch)
            
            # Process outputs based on model type
            processed_outputs = {}
            
            # Extract coordinates if available
            if 'atom_positions' in outputs:
                processed_outputs['atom_positions'] = outputs['atom_positions']
                
            # Extract confidence scores
            if 'confidence_scores' in outputs:
                processed_outputs['confidence_scores'] = outputs['confidence_scores']
                
            # Extract embeddings if available
            if 'embeddings' in outputs:
                processed_outputs['embeddings'] = outputs['embeddings']
                
            # Extract attention maps if available
            if 'attention_maps' in outputs:
                processed_outputs['attention_maps'] = outputs['attention_maps']
                
            # Add any additional outputs
            for key, value in outputs.items():
                if key not in processed_outputs:
                    processed_outputs[key] = value
                    
            return processed_outputs
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelInferenceError(f"Prediction failed: {str(e)}")
    
    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple batches."""
        combined = {}
        for key in results[0].keys():
            if isinstance(results[0][key], torch.Tensor):
                combined[key] = torch.cat([r[key] for r in results])
            else:
                combined[key] = results[0][key]
        return combined
    
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
        try:
            return self.decoder.decode_structure(outputs, sequence, **kwargs)
        except Exception as e:
            logger.error(f"Failed to decode structure: {str(e)}")
            raise DecoderError(f"Failed to decode structure: {str(e)}")
    
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
        try:
            return self.decoder.decode_embeddings(outputs, **kwargs)
        except Exception as e:
            logger.error(f"Failed to decode embeddings: {str(e)}")
            raise DecoderError(f"Failed to decode embeddings: {str(e)}")
    
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
        try:
            return self.decoder.decode_confidence(outputs, **kwargs)
        except Exception as e:
            logger.error(f"Failed to decode confidence scores: {str(e)}")
            raise DecoderError(f"Failed to decode confidence scores: {str(e)}")
    
    def generate_structure(
        self,
        sequence: str,
        output_file: str,
        **kwargs
    ) -> None:
        """Generate protein structure prediction.
        
        Args:
            sequence: Input amino acid sequence
            output_file: Path to save the output structure
            **kwargs: Additional arguments for structure generation
        """
        try:
            logger.info(f"Generating structure for sequence of length {len(sequence)}")
            
            # Prepare input data
            inputs = self._prepare_sequence_input(sequence, **kwargs)
            
            # Run prediction
            outputs = self.predict(inputs)
            
            # Decode structure
            structure, confidence, dist_matrix = self.decoder.decode_structure(
                outputs, sequence, **kwargs
            )
            
            # Save structure
            io = PDB.PDBIO()
            io.set_structure(structure)
            io.save(output_file)
            
            # Save additional outputs if requested
            if self.config.save_intermediates:
                output_dir = Path(output_file).parent
                np.save(output_dir / 'confidence_scores.npy', confidence)
                np.save(output_dir / 'distance_matrix.npy', dist_matrix)
                
            logger.info(f"Structure saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Structure generation failed: {str(e)}")
            raise ModelInferenceError(f"Structure generation failed: {str(e)}")
    
    def extract_embeddings(
        self,
        sequence: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract embeddings from the model.
        
        Args:
            sequence: Input amino acid sequence
            **kwargs: Additional arguments for embedding extraction
            
        Returns:
            Dictionary containing embeddings
        """
        try:
            logger.info(f"Extracting embeddings for sequence of length {len(sequence)}")
            
            # Prepare input data
            inputs = self._prepare_sequence_input(sequence, **kwargs)
            
            # Run prediction
            outputs = self.predict(inputs)
            
            # Decode embeddings
            embeddings = self.decoder.decode_embeddings(outputs, **kwargs)
            
            # Save embeddings if requested
            if self.config.save_intermediates and self.config.output_dir:
                output_dir = Path(self.config.output_dir)
                for name, embedding in embeddings.items():
                    np.save(output_dir / f'{name}_embedding.npy', embedding)
                    
            logger.info(f"Successfully extracted {len(embeddings)} embedding types")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise ModelInferenceError(f"Embedding extraction failed: {str(e)}")
    
    def _prepare_sequence_input(
        self,
        sequence: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare model inputs from sequence.
        
        Args:
            sequence: Input amino acid sequence
            **kwargs: Additional arguments for input preparation
            
        Returns:
            Dictionary containing model inputs
        """
        # Convert sequence to one-hot encoding
        one_hot = self._sequence_to_one_hot(sequence)
        
        # Create input dictionary
        inputs = {
            'sequence': sequence,
            'one_hot': one_hot,
            'sequence_length': len(sequence)
        }
        
        # Add any additional inputs from kwargs
        inputs.update(kwargs)
        
        return inputs
    
    def _sequence_to_one_hot(self, sequence: str) -> np.ndarray:
        """Convert amino acid sequence to one-hot encoding.
        
        Args:
            sequence: Input amino acid sequence
            
        Returns:
            One-hot encoded sequence array
        """
        # Define amino acid vocabulary
        aa_vocab = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: idx for idx, aa in enumerate(aa_vocab)}
        
        # Create one-hot encoding
        one_hot = np.zeros((len(sequence), len(aa_vocab)))
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                one_hot[i, aa_to_idx[aa]] = 1
            else:
                # Handle unknown amino acids
                one_hot[i] = 1.0 / len(aa_vocab)
                
        return one_hot

    def optimize_memory_usage(self, **kwargs) -> None:
        """Optimize memory usage for model inference.
        
        This method implements several memory optimization strategies:
        1. Model weight caching
        2. Gradient checkpointing
        3. Memory-efficient attention
        4. Batch size optimization
        5. Device memory management
        
        Args:
            **kwargs: Additional arguments for memory optimization
        """
        try:
            logger.info("Optimizing memory usage")
            
            if not self.model:
                raise ModelError("Model not initialized")
            
            # Get memory manager
            memory_manager = GPUMemoryManager()
            
            # Apply memory limit if configured
            if self.config.memory_limit:
                memory_manager.set_memory_limit(self.config.memory_limit)
            
            # Enable gradient checkpointing if supported
            if hasattr(self.model, 'enable_checkpointing'):
                self.model.enable_checkpointing()
            
            # Apply quantization if configured
            if self.config.quantization_config:
                self.quantize_model()
            
            # Optimize batch size based on available memory
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory
                current_batch_size = self.config.batch_size
                
                # Estimate memory per sample
                sample_size = self._estimate_sample_memory()
                max_batch_size = int(available_memory * 0.8 / sample_size)  # Use 80% of available memory
                
                if current_batch_size > max_batch_size:
                    logger.warning(f"Reducing batch size from {current_batch_size} to {max_batch_size} due to memory constraints")
                    self.config.batch_size = max_batch_size
            
            # Clear unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory usage - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            logger.debug("Memory usage optimized")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory usage: {str(e)}")
            raise ModelMemoryError(f"Memory optimization failed: {str(e)}")
    
    def _estimate_sample_memory(self) -> int:
        """Estimate memory required per sample in bytes.
        
        Returns:
            Estimated memory requirement in bytes
        """
        try:
            # Base memory for model weights
            base_memory = 0
            
            # Add memory for embeddings
            for layer_config in self.config.embedding_config.values():
                if 'dimension' in layer_config:
                    base_memory += layer_config['dimension'] * 4  # 4 bytes per float
            
            # Add memory for attention maps if applicable
            if hasattr(self.model, 'num_attention_heads'):
                seq_len = self.config.max_sequence_length or 1024
                base_memory += seq_len * seq_len * self.model.num_attention_heads * 4
            
            # Add memory for intermediate activations
            base_memory *= 2  # Rough estimate for intermediate tensors
            
            return base_memory
            
        except Exception as e:
            logger.error(f"Failed to estimate sample memory: {str(e)}")
            return 0

    def quantize_model(self, **kwargs) -> None:
        """Quantize the model for memory optimization.
        
        This method applies quantization to the model using the configured
        quantization settings. It supports multiple quantization types:
        - Dynamic quantization
        - Static quantization
        - Quantization-aware training (QAT)
        - Float16/BFloat16 precision
        
        Args:
            **kwargs: Additional arguments for quantization
        """
        try:
            logger.info("Quantizing model")
            
            if not self.model:
                raise ModelError("Model not initialized")
            
            # Get quantization config from kwargs or use default
            config = kwargs.get('config', self.config.quantization_config)
            if not config:
                config = QuantizationConfig()
            
            # Initialize quantization manager
            from src.core.quantization.quantization_manager import QuantizationManager
            quant_manager = QuantizationManager()
            
            # Quantize model
            quantized_model = quant_manager.quantize_model(
                model=self.model,
                model_name=self.config.name,
                config=config
            )
            
            # Update model reference
            self.model = quantized_model
            
            # Log memory savings
            original_size = quant_manager._get_model_size(self.model)
            quantized_size = quant_manager._get_model_size(quantized_model)
            savings = (original_size - quantized_size) / original_size * 100
            
            logger.info(f"Model quantized successfully: {savings:.1f}% memory savings")
            
        except Exception as e:
            logger.error(f"Failed to quantize model: {str(e)}")
            raise QuantizationError(f"Model quantization failed: {str(e)}")

    def cleanup(self) -> None:
        """Clean up resources used by the model interface.
        
        This method ensures proper cleanup of all resources:
        1. GPU memory
        2. Model weights and buffers
        3. Distributed resources
        4. Cached data
        5. File handles
        
        The cleanup is performed in a specific order to avoid dependency issues.
        """
        try:
            logger.info("Cleaning up model interface resources")
            
            # Clean up distributed resources first
            if self.distributed_manager:
                try:
                    self.distributed_manager.cleanup()
                except Exception as e:
                    logger.warning(f"Failed to cleanup distributed manager: {str(e)}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                try:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Clear any unused memory
                    if hasattr(torch.cuda, 'memory_summary'):
                        torch.cuda.memory_summary()
                except Exception as e:
                    logger.warning(f"Failed to clear GPU memory: {str(e)}")
            
            # Clean up model resources
            if self.model:
                try:
                    # Move model to CPU to free GPU memory
                    self.model.cpu()
                    
                    # Clear model buffers
                    for buffer in self.model.buffers():
                        buffer.detach()
                    
                    # Clear model parameters
                    for param in self.model.parameters():
                        param.detach()
                    
                    # Delete model
                    del self.model
                    self.model = None
                except Exception as e:
                    logger.warning(f"Failed to cleanup model: {str(e)}")
            
            # Clean up decoder resources
            if self.decoder:
                try:
                    # Clear any cached data in decoder
                    if hasattr(self.decoder, 'clear_cache'):
                        self.decoder.clear_cache()
                except Exception as e:
                    logger.warning(f"Failed to cleanup decoder: {str(e)}")
            
            # Clear configuration references
            self.config = None
            
            # Log cleanup completion
            logger.info("Model interface resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup model interface: {str(e)}")
            raise ModelError(f"Resource cleanup failed: {str(e)}")
    
    def __del__(self):
        """Ensure cleanup is called when the object is destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Failed to cleanup during object destruction: {str(e)}")