import torch
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Types of quantization supported."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization-Aware Training
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    type: QuantizationType
    dtype: Optional[torch.dtype] = None
    calibration_data: Optional[Any] = None
    num_calibration_batches: int = 100
    per_channel: bool = True
    symmetric: bool = True
    preserve_accuracy: bool = True

class QuantizationManager:
    """
    Manages model quantization for memory optimization.
    
    This class provides utilities for quantizing models using various methods
    while preserving model accuracy as much as possible.
    """
    
    def __init__(self):
        """Initialize the quantization manager."""
        self.quantized_models: Dict[str, torch.nn.Module] = {}
        self.original_models: Dict[str, torch.nn.Module] = {}
    
    def quantize_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        config: QuantizationConfig
    ) -> torch.nn.Module:
        """
        Quantize a model using the specified configuration.
        
        Args:
            model: Model to quantize
            model_name: Name of the model
            config: Quantization configuration
            
        Returns:
            Quantized model
        """
        try:
            # Store original model
            self.original_models[model_name] = model
            
            if config.type == QuantizationType.DYNAMIC:
                quantized_model = self._dynamic_quantization(model, config)
            elif config.type == QuantizationType.STATIC:
                quantized_model = self._static_quantization(model, config)
            elif config.type == QuantizationType.QAT:
                quantized_model = self._qat_quantization(model, config)
            elif config.type == QuantizationType.FLOAT16:
                quantized_model = self._float16_quantization(model)
            elif config.type == QuantizationType.BFLOAT16:
                quantized_model = self._bfloat16_quantization(model)
            else:
                raise ValueError(f"Unsupported quantization type: {config.type}")
            
            # Store quantized model
            self.quantized_models[model_name] = quantized_model
            
            # Log memory savings
            original_size = self._get_model_size(model)
            quantized_size = self._get_model_size(quantized_model)
            savings = (original_size - quantized_size) / original_size * 100
            
            logger.info(f"Quantized {model_name}: {savings:.1f}% memory savings")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Failed to quantize model {model_name}: {str(e)}")
            raise
    
    def _dynamic_quantization(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig
    ) -> torch.nn.Module:
        """Apply dynamic quantization to the model."""
        try:
            # Prepare model for quantization
            model.eval()
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=config.dtype or torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {str(e)}")
            raise
    
    def _static_quantization(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig
    ) -> torch.nn.Module:
        """Apply static quantization to the model."""
        try:
            # Prepare model for quantization
            model.eval()
            
            # Prepare quantization configuration
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=config.dtype or torch.qint8,
                    qscheme=torch.per_tensor_symmetric if config.symmetric else torch.per_tensor_affine
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=config.dtype or torch.qint8,
                    qscheme=torch.per_channel_symmetric if config.per_channel else torch.per_tensor_symmetric
                )
            )
            
            # Prepare model for quantization
            model.qconfig = qconfig
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate model if calibration data is provided
            if config.calibration_data is not None:
                self._calibrate_model(model, config.calibration_data, config.num_calibration_batches)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=True)
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Static quantization failed: {str(e)}")
            raise
    
    def _qat_quantization(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig
    ) -> torch.nn.Module:
        """Apply quantization-aware training to the model."""
        try:
            # Prepare model for QAT
            model.train()
            
            # Prepare quantization configuration
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    dtype=config.dtype or torch.qint8,
                    qscheme=torch.per_tensor_symmetric if config.symmetric else torch.per_tensor_affine
                ),
                weight=torch.quantization.FakeQuantize.with_args(
                    dtype=config.dtype or torch.qint8,
                    qscheme=torch.per_channel_symmetric if config.per_channel else torch.per_tensor_symmetric
                )
            )
            
            # Prepare model for QAT
            model.qconfig = qconfig
            torch.quantization.prepare_qat(model, inplace=True)
            
            return model
            
        except Exception as e:
            logger.error(f"QAT quantization failed: {str(e)}")
            raise
    
    def _float16_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Convert model to float16 precision."""
        try:
            return model.half()
        except Exception as e:
            logger.error(f"Float16 quantization failed: {str(e)}")
            raise
    
    def _bfloat16_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Convert model to bfloat16 precision."""
        try:
            return model.to(torch.bfloat16)
        except Exception as e:
            logger.error(f"Bfloat16 quantization failed: {str(e)}")
            raise
    
    def _calibrate_model(
        self,
        model: torch.nn.Module,
        calibration_data: Any,
        num_batches: int
    ) -> None:
        """Calibrate model for static quantization."""
        try:
            with torch.no_grad():
                for i in range(num_batches):
                    if i >= len(calibration_data):
                        break
                    model(calibration_data[i])
        except Exception as e:
            logger.error(f"Model calibration failed: {str(e)}")
            raise
    
    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Get model size in bytes."""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return param_size + buffer_size
        except Exception as e:
            logger.error(f"Failed to get model size: {str(e)}")
            return 0
    
    def restore_original_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Restore the original model from quantization.
        
        Args:
            model_name: Name of the model to restore
            
        Returns:
            Original model if available, None otherwise
        """
        return self.original_models.get(model_name)
    
    def get_quantized_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Get the quantized version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Quantized model if available, None otherwise
        """
        return self.quantized_models.get(model_name)

    def dynamic_precision_adjustment(self):
        """Dynamically adjust precision based on requirements"""
        pass
    
    def validate_quantization_impact(
        self,
        model: torch.nn.Module,
        validation_data: Dict[str, torch.Tensor],
        metrics: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> Dict[str, float]:
        """Validate impact of quantization on model accuracy.
        
        Args:
            model: The quantized model to validate
            validation_data: Dictionary containing validation inputs and targets
            metrics: List of metrics to evaluate (e.g., ['accuracy', 'mse', 'mae'])
            threshold: Maximum allowed relative performance degradation
            
        Returns:
            Dictionary containing validation metrics
        """
        try:
            logger.info("Validating quantization impact")
            
            # Default metrics if none provided
            metrics = metrics or ['accuracy']
            
            # Get original model predictions
            with torch.no_grad():
                original_outputs = self.original_model(**validation_data)
                
            # Get quantized model predictions
            with torch.no_grad():
                quantized_outputs = model(**validation_data)
                
            # Calculate metrics
            results = {}
            
            if 'accuracy' in metrics:
                orig_acc = self._calculate_accuracy(
                    original_outputs, validation_data['targets']
                )
                quant_acc = self._calculate_accuracy(
                    quantized_outputs, validation_data['targets']
                )
                
                relative_drop = (orig_acc - quant_acc) / orig_acc
                results['accuracy'] = {
                    'original': orig_acc,
                    'quantized': quant_acc,
                    'relative_drop': relative_drop
                }
                
            if 'mse' in metrics:
                mse_increase = self._calculate_mse_increase(
                    original_outputs, quantized_outputs
                )
                results['mse'] = mse_increase
                
            if 'mae' in metrics:
                mae_increase = self._calculate_mae_increase(
                    original_outputs, quantized_outputs
                )
                results['mae'] = mae_increase
                
            # Check if any metric exceeds threshold
            validation_failed = False
            failure_metrics = []
            
            if 'accuracy' in results:
                if results['accuracy']['relative_drop'] > threshold:
                    validation_failed = True
                    failure_metrics.append(
                        f"accuracy drop: {results['accuracy']['relative_drop']:.3f}"
                    )
                    
            if 'mse' in results and results['mse'] > threshold:
                validation_failed = True
                failure_metrics.append(f"MSE increase: {results['mse']:.3f}")
                
            if 'mae' in results and results['mae'] > threshold:
                validation_failed = True
                failure_metrics.append(f"MAE increase: {results['mae']:.3f}")
                
            if validation_failed:
                raise ValueError(
                    f"Quantization validation failed. Metrics exceeding threshold: "
                    f"{', '.join(failure_metrics)}"
                )
                
            logger.info("Quantization validation successful")
            return results
            
        except Exception as e:
            logger.error(f"Quantization validation failed: {str(e)}")
            raise ModelError(f"Quantization validation failed: {str(e)}")
    
    def _calculate_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Calculate prediction accuracy."""
        predictions = outputs.argmax(dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total
    
    def _calculate_mse_increase(
        self,
        original_outputs: torch.Tensor,
        quantized_outputs: torch.Tensor
    ) -> float:
        """Calculate relative increase in MSE."""
        original_mse = torch.nn.functional.mse_loss(
            original_outputs, quantized_outputs
        ).item()
        return original_mse
    
    def _calculate_mae_increase(
        self,
        original_outputs: torch.Tensor,
        quantized_outputs: torch.Tensor
    ) -> float:
        """Calculate relative increase in MAE."""
        original_mae = torch.nn.functional.l1_loss(
            original_outputs, quantized_outputs
        ).item()
        return original_mae