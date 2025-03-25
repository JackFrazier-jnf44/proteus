"""Distributed computing management utilities."""

import torch
import torch.distributed as dist
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from src.exceptions import ModelError
from src.core.memory import GPUMemoryManager

logger = logging.getLogger(__name__)

@dataclass
class GPUStats:
    """GPU statistics for workload distribution."""
    device_id: int
    total_memory: int
    used_memory: int
    compute_capability: Tuple[int, int]
    utilization: float

def optimize_multi_gpu_distribution(
    model_size: int,
    batch_size: int,
    available_gpus: Optional[List[int]] = None
) -> Dict[int, int]:
    """Optimize workload across multiple GPUs.
    
    Args:
        model_size: Size of model in bytes
        batch_size: Total batch size to distribute
        available_gpus: List of available GPU device IDs
        
    Returns:
        Dictionary mapping GPU IDs to their assigned batch sizes
    """
    try:
        if not torch.cuda.is_available():
            raise ModelError("No CUDA devices available")
            
        # Get available GPUs
        if available_gpus is None:
            available_gpus = list(range(torch.cuda.device_count()))
            
        if not available_gpus:
            raise ModelError("No GPUs specified for distribution")
            
        # Collect GPU statistics
        gpu_stats = []
        for gpu_id in available_gpus:
            memory = torch.cuda.get_device_properties(gpu_id).total_memory
            used = torch.cuda.memory_allocated(gpu_id)
            capability = torch.cuda.get_device_capability(gpu_id)
            
            # Get GPU utilization (requires nvidia-smi)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100.0
            except:
                utilization = 0.0
                
            gpu_stats.append(GPUStats(
                device_id=gpu_id,
                total_memory=memory,
                used_memory=used,
                compute_capability=capability,
                utilization=utilization
            ))
            
        # Calculate memory requirements per sample
        memory_per_sample = model_size / batch_size
        
        # Calculate available memory per GPU
        available_memory = {
            gpu.device_id: gpu.total_memory - gpu.used_memory
            for gpu in gpu_stats
        }
        
        # Calculate initial distribution based on available memory
        total_available = sum(available_memory.values())
        distribution = {}
        remaining_batch = batch_size
        
        for gpu_id, memory in available_memory.items():
            # Calculate batch size proportion based on available memory
            gpu_batch = int(batch_size * (memory / total_available))
            
            # Adjust for GPU compute capability and utilization
            gpu_stats_obj = next(g for g in gpu_stats if g.device_id == gpu_id)
            capability_factor = sum(gpu_stats_obj.compute_capability) / 10.0
            utilization_factor = 1.0 - gpu_stats_obj.utilization
            
            adjusted_batch = int(gpu_batch * capability_factor * utilization_factor)
            
            # Ensure minimum batch size of 1
            distribution[gpu_id] = max(1, min(adjusted_batch, remaining_batch))
            remaining_batch -= distribution[gpu_id]
            
        # Distribute any remaining samples
        if remaining_batch > 0:
            sorted_gpus = sorted(
                gpu_stats,
                key=lambda x: (
                    x.total_memory - x.used_memory,
                    sum(x.compute_capability),
                    -x.utilization
                ),
                reverse=True
            )
            
            for gpu in sorted_gpus:
                if remaining_batch <= 0:
                    break
                    
                # Add one sample at a time to the most suitable GPU
                distribution[gpu.device_id] += 1
                remaining_batch -= 1
                
        logger.info(
            f"Optimized batch distribution across {len(distribution)} GPUs: "
            f"{distribution}"
        )
        return distribution
        
    except Exception as e:
        logger.error(f"Failed to optimize GPU distribution: {str(e)}")
        raise ModelError(f"GPU distribution optimization failed: {str(e)}")

def handle_node_failures(
    active_nodes: List[str],
    failed_nodes: List[str],
    model_weights: Dict[str, torch.Tensor],
    checkpoint_dir: Optional[Path] = None
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Handle node failures and recovery.
    
    Args:
        active_nodes: List of currently active node addresses
        failed_nodes: List of failed node addresses
        model_weights: Current model weight distribution
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        Tuple containing:
        - Updated list of active nodes
        - Updated model weight distribution
    """
    try:
        if not failed_nodes:
            return active_nodes, model_weights
            
        logger.warning(f"Handling failure of nodes: {failed_nodes}")
        
        # Remove failed nodes from active list
        active_nodes = [node for node in active_nodes if node not in failed_nodes]
        
        if not active_nodes:
            raise ModelError("No active nodes remaining after failures")
            
        # Save checkpoint if directory provided
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir / "failure_checkpoint.pt"
            torch.save({
                'model_weights': model_weights,
                'active_nodes': active_nodes,
                'failed_nodes': failed_nodes
            }, checkpoint_path)
            logger.info(f"Saved failure checkpoint to {checkpoint_path}")
            
        # Redistribute weights from failed nodes
        redistributed_weights = {}
        weights_per_node = len(model_weights) // len(active_nodes)
        
        for i, (key, tensor) in enumerate(model_weights.items()):
            target_node = active_nodes[i % len(active_nodes)]
            if target_node not in redistributed_weights:
                redistributed_weights[target_node] = {}
            redistributed_weights[target_node][key] = tensor
            
        # Update process group
        if dist.is_initialized():
            dist.destroy_process_group()
            
        # Reinitialize with remaining nodes
        dist.init_process_group(
            backend='nccl',
            world_size=len(active_nodes),
            rank=active_nodes.index(dist.get_rank())
        )
        
        logger.info(
            f"Successfully redistributed workload to {len(active_nodes)} "
            f"remaining nodes"
        )
        return active_nodes, redistributed_weights
        
    except Exception as e:
        logger.error(f"Failed to handle node failures: {str(e)}")
        raise ModelError(f"Node failure handling failed: {str(e)}") 