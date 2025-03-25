"""Distributed inference support with user-defined weights."""

import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.core.memory.memory_management import GPUMemoryManager
from src.core.memory.model_caching import ModelCache
from src.interfaces.base_interface import BaseModelInterface, BaseModelConfig
from src.exceptions import ModelError, ModelInitializationError, ModelInferenceError

logger = logging.getLogger(__name__)

class DistributionStrategy(Enum):
    """Strategy for distributing model weights across devices."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    CUSTOM = "custom"

@dataclass
class DistributedConfig:
    """Configuration for distributed inference."""
    strategy: DistributionStrategy
    weights: Optional[Dict[str, float]] = None
    devices: Optional[List[str]] = None
    batch_size: int = 1
    sync_frequency: int = 10
    use_gpu: bool = True
    memory_limit: Optional[int] = None
    layer_rules: Optional[Dict[str, Dict[str, Any]]] = None
    sync_timeout: float = 30.0  # Timeout for synchronization in seconds

class DistributedInferenceManager:
    """Manages distributed inference across multiple devices."""
    
    def __init__(
        self,
        base_model: BaseModelInterface,
        config: DistributedConfig,
        weight_cache: Optional[ModelCache] = None
    ):
        """Initialize distributed inference manager."""
        self.base_model = base_model
        self.config = config
        self.weight_cache = weight_cache or ModelCache()
        self.devices = self._get_available_devices()
        self.models: Dict[str, BaseModelInterface] = {}
        self.memory_managers: Dict[str, GPUMemoryManager] = {}
        self.sync_barrier = None
        
        try:
            if self.config.use_gpu and not torch.cuda.is_available():
                raise ModelInitializationError("GPU not available")
            dist.init_process_group(backend='nccl' if self.config.use_gpu else 'gloo')
            logger.info(f"Initialized distributed environment with {len(self.devices)} devices")
            
            # Create synchronization barrier
            self.sync_barrier = dist.Barrier()
            
            # Distribute model weights
            weights = self.base_model.model.state_dict()
            weight_splits = self._split_weights(weights)
            
            for device, device_weights in zip(self.devices, weight_splits):
                device_config = BaseModelConfig(
                    name=f"{self.base_model.config.name}_{device}",
                    model_type=self.base_model.config.model_type,
                    output_format=self.base_model.config.output_format,
                    embedding_config=self.base_model.config.embedding_config,
                    device=device,
                    memory_limit=self.config.memory_limit
                )
                
                model = self._create_device_model(device_config, device_weights)
                self.models[device] = model
                self.memory_managers[device] = GPUMemoryManager(
                    memory_limit=self.config.memory_limit,
                    device=device
                )
            
            logger.info(f"Successfully distributed model across {len(self.devices)} devices")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {str(e)}")
            raise ModelInitializationError(f"Distributed initialization failed: {str(e)}")
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available devices."""
        return (self.config.devices or 
                [f"cuda:{i}" for i in range(torch.cuda.device_count())] if self.config.use_gpu 
                else ["cpu"])
    
    def _split_weights(self, weights: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split model weights based on distribution strategy."""
        strategy_map = {
            DistributionStrategy.ROUND_ROBIN: self._round_robin_split,
            DistributionStrategy.WEIGHTED: self._weighted_split,
            DistributionStrategy.CUSTOM: self._custom_split
        }
        
        if self.config.strategy not in strategy_map:
            raise ValueError(f"Unsupported distribution strategy: {self.config.strategy}")
        
        return strategy_map[self.config.strategy](weights)
    
    def _round_robin_split(self, weights: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split weights using round-robin distribution."""
        splits = [{} for _ in self.devices]
        for i, (key, tensor) in enumerate(weights.items()):
            splits[i % len(self.devices)][key] = tensor
        return splits
    
    def _weighted_split(self, weights: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split weights using weighted distribution."""
        try:
            if not self.config.weights:
                logger.warning("No weights provided for weighted distribution, falling back to round-robin")
                return self._round_robin_split(weights)
            
            if not all(isinstance(w, (int, float)) and w > 0 for w in self.config.weights.values()):
                raise ValueError("All weights must be positive numbers")
            
            if len(self.config.weights) != len(self.devices):
                raise ValueError(f"Number of weights ({len(self.config.weights)}) must match number of devices ({len(self.devices)})")
            
            normalized_weights = {k: v/sum(self.config.weights.values()) 
                                for k, v in self.config.weights.items()}
            cumulative_weights = [sum(list(normalized_weights.values())[:i+1]) 
                                for i in range(len(normalized_weights))]
            
            splits = [{} for _ in self.devices]
            layer_groups = self._group_weights_by_layer(weights)
            
            for layer_weights in layer_groups.values():
                for key, tensor in layer_weights.items():
                    device_idx = self._find_optimal_device(
                        key, tensor, normalized_weights, cumulative_weights, splits
                    )
                    splits[device_idx][key] = tensor
            
            self._log_distribution_stats(splits, normalized_weights)
            return splits
            
        except Exception as e:
            logger.error(f"Failed to perform weighted split: {str(e)}")
            raise ModelInitializationError(f"Weighted split failed: {str(e)}")
    
    def _custom_split(self, weights: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split weights using custom distribution strategy."""
        try:
            if not self.config.layer_rules:
                raise ValueError("Layer rules must be provided for custom distribution")
            
            splits = [{} for _ in self.devices]
            layer_groups = self._group_weights_by_layer(weights)
            
            for layer_type, layer_weights in layer_groups.items():
                layer_rules = self._get_layer_distribution_rules(layer_type)
                for key, tensor in layer_weights.items():
                    device_idx = self._determine_target_device(key, tensor, layer_rules, splits)
                    splits[device_idx][key] = tensor
            
            self._log_distribution_stats(splits, {device: 1.0/len(self.devices) for device in self.devices})
            return splits
            
        except Exception as e:
            logger.error(f"Failed to perform custom split: {str(e)}")
            raise ModelInitializationError(f"Custom split failed: {str(e)}")
    
    def _group_weights_by_layer(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Group weights by layer type."""
        layer_groups = {}
        for key, tensor in weights.items():
            layer_type = key.split('.')[0]
            if layer_type not in layer_groups:
                layer_groups[layer_type] = {}
            layer_groups[layer_type][key] = tensor
        return layer_groups
    
    def _get_layer_distribution_rules(self, layer_type: str) -> Dict[str, Any]:
        """Get distribution rules for a layer type."""
        default_rules = {
            'device_weights': {device: 1.0 for device in self.devices},
            'memory_threshold': 0.8,
            'preferred_devices': self.devices
        }
        return ({**default_rules, **self.config.layer_rules[layer_type]}
                if hasattr(self.config, 'layer_rules') and layer_type in self.config.layer_rules
                else default_rules)
    
    def _find_optimal_device(
        self,
        key: str,
        tensor: torch.Tensor,
        normalized_weights: Dict[str, float],
        cumulative_weights: List[float],
        splits: List[Dict[str, torch.Tensor]]
    ) -> int:
        """Find optimal device for a weight tensor."""
        device_memory = [
            sum(t.numel() * t.element_size() for t in device_weights.values())
            for device_weights in splits
        ]
        
        min_memory = float('inf')
        optimal_device = 0
        
        for i, (device, weight) in enumerate(zip(self.devices, normalized_weights.values())):
            if device_memory[i] < min_memory and device_memory[i] < min_memory * 1.1:
                min_memory = device_memory[i]
                optimal_device = i
        
        return optimal_device
    
    def _determine_target_device(
        self,
        key: str,
        tensor: torch.Tensor,
        rules: Dict[str, Any],
        splits: List[Dict[str, torch.Tensor]]
    ) -> int:
        """Determine target device based on rules."""
        device_memory = [
            sum(t.numel() * t.element_size() for t in device_weights.values())
            for device_weights in splits
        ]
        
        min_memory = float('inf')
        optimal_device = 0
        
        for i, device in enumerate(self.devices):
            if (device in rules['preferred_devices'] and 
                device_memory[i] / self.config.memory_limit <= rules['memory_threshold'] and
                device_memory[i] < min_memory):
                min_memory = device_memory[i]
                optimal_device = i
        
        return optimal_device if min_memory != float('inf') else device_memory.index(min(device_memory))
    
    def _log_distribution_stats(
        self,
        splits: List[Dict[str, torch.Tensor]],
        normalized_weights: Dict[str, float]
    ) -> None:
        """Log distribution statistics."""
        total_size = sum(
            sum(tensor.numel() * tensor.element_size() for tensor in device_weights.values())
            for device_weights in splits
        )
        
        for i, (device, device_weights) in enumerate(zip(self.devices, splits)):
            device_size = sum(tensor.numel() * tensor.element_size() 
                            for tensor in device_weights.values())
            percentage = (device_size / total_size) * 100
            target_percentage = normalized_weights[device] * 100
            
            logger.debug(
                f"Device {device}: {percentage:.1f}% of weights "
                f"(target: {target_percentage:.1f}%)"
            )
    
    def _create_device_model(
        self,
        config: BaseModelConfig,
        weights: Dict[str, torch.Tensor]
    ) -> BaseModelInterface:
        """Create model instance for a device."""
        try:
            model = self.base_model.__class__(config)
            model.model.load_state_dict(weights)
            model.model.to(config.device)
            return model
        except Exception as e:
            logger.error(f"Failed to create device model: {str(e)}")
            raise ModelInitializationError(f"Device model creation failed: {str(e)}")
    
    def predict(
        self,
        inputs: Dict[str, Any],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run distributed prediction."""
        try:
            batch_size = batch_size or self.config.batch_size
            batches = self._split_batches(inputs, batch_size)
            
            results = {}
            for device, model in self.models.items():
                device_results = []
                for batch in batches:
                    device_batch = self._move_to_device(batch, device)
                    with torch.no_grad():
                        batch_result = model.predict(device_batch)
                    device_results.append(batch_result)
                results[device] = self._combine_results(device_results)
            
            return self._synchronize_results(results)
            
        except Exception as e:
            logger.error(f"Distributed prediction failed: {str(e)}")
            raise ModelInferenceError(f"Distributed prediction failed: {str(e)}")
    
    def _split_batches(self, inputs: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
        """Split inputs into batches."""
        num_samples = len(next(iter(inputs.values())))
        return [
            {k: v[i:i+batch_size] if isinstance(v, torch.Tensor) else v
             for k, v in inputs.items()}
            for i in range(0, num_samples, batch_size)
        ]
    
    def _move_to_device(self, batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """Move batch to device."""
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
    
    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from batches."""
        return {
            k: torch.cat([r[k] for r in results]) if isinstance(results[0][k], torch.Tensor)
            else results[0][k]
            for k in results[0].keys()
        }
    
    def _synchronize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize results across devices."""
        try:
            # Wait for all processes to reach synchronization point
            self.sync_barrier.wait()
            
            ref_device = list(results.keys())[0]
            ref_results = results[ref_device]
            synchronized = {}
            
            for key, ref_value in ref_results.items():
                if isinstance(ref_value, torch.Tensor):
                    synchronized[key] = torch.cat([
                        results[device][key].to(device)
                        for device in self.devices
                        if device in results and key in results[device]
                    ])
                elif isinstance(ref_value, np.ndarray):
                    synchronized[key] = torch.cat([
                        torch.from_numpy(results[device][key]).to(device)
                        for device in self.devices
                        if device in results and key in results[device]
                    ]).cpu().numpy()
                elif isinstance(ref_value, (list, tuple)):
                    synchronized[key] = type(ref_value)(
                        self._synchronize_results({
                            device: results[device][key][i]
                            for device in results.keys()
                            for i in range(len(results[device][key]))
                        })
                        for i in range(len(ref_value))
                    )
                elif isinstance(ref_value, dict):
                    synchronized[key] = {
                        k: self._synchronize_results({
                            device: results[device][key][k]
                            for device in results.keys()
                        })
                        for k in ref_value.keys()
                    }
                else:
                    synchronized[key] = ref_value
            
            # Wait for all processes to complete synchronization
            self.sync_barrier.wait()
            
            logger.debug(f"Synchronized {len(synchronized)} result keys across {len(self.devices)} devices")
            return synchronized
            
        except Exception as e:
            logger.error(f"Failed to synchronize results: {str(e)}")
            raise ModelInferenceError(f"Result synchronization failed: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up distributed resources."""
        try:
            for model in self.models.values():
                model.clear_gpu_memory()
            for manager in self.memory_managers.values():
                manager.clear_memory()
            dist.destroy_process_group()
            logger.info("Successfully cleaned up distributed resources")
        except Exception as e:
            logger.error(f"Failed to clean up distributed resources: {str(e)}")
            raise ModelError(f"Cleanup failed: {str(e)}") 