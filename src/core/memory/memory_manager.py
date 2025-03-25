"""Memory management utilities for large-scale predictions."""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import gc
import psutil
import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.exceptions import MemoryError, ModelMemoryError, ResourceError

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Statistics for memory usage tracking."""
    total_memory: int = 0
    allocated_memory: int = 0
    peak_memory: int = 0
    last_peak_time: datetime = field(default_factory=datetime.now)
    memory_history: List[Dict[str, Any]] = field(default_factory=list)
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_gpu_memory: Optional[float] = None  # Maximum GPU memory to use (GB)
    max_cpu_memory: Optional[float] = None  # Maximum CPU memory to use (GB)
    chunk_size: int = 1024  # Size of chunks for processing
    offload_threshold: float = 0.8  # Memory usage threshold for offloading
    enable_attention_offloading: bool = True  # Enable attention offloading
    enable_template_offloading: bool = True  # Enable template offloading
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    pin_memory: bool = True  # Pin memory for faster CPU-GPU transfers
    max_cache_size: Optional[float] = None  # Maximum size for model weight cache (GB)

class MemoryManager:
    """Unified memory management system for large-scale predictions."""
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize memory manager.
        
        Args:
            config: Memory configuration
            device: Device to manage memory for
            cache_dir: Optional directory for storing cached weights
        """
        self.config = config or MemoryConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize memory tracking
        self.stats = MemoryStats()
        self._initialize_stats()
        
        # Set up memory limits
        if self.device == "cuda":
            if self.config.max_gpu_memory is None:
                self.config.max_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.cuda.set_per_process_memory_fraction(
                self.config.max_gpu_memory / torch.cuda.get_device_properties(0).total_memory * 1e9
            )
        
        if self.config.max_cpu_memory is None:
            self.config.max_cpu_memory = psutil.virtual_memory().total / 1e9
        
        # Initialize weight cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.model_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = {}
        self._load_cache_index()
        
        logger.info(f"Initialized memory manager for {self.device}")
    
    def _initialize_stats(self) -> None:
        """Initialize memory statistics."""
        try:
            if torch.cuda.is_available():
                self.stats.total_memory = torch.cuda.get_device_properties(0).total_memory
                self.stats.allocated_memory = torch.cuda.memory_allocated(0)
                self.stats.peak_memory = torch.cuda.max_memory_allocated(0)
            else:
                self.stats.total_memory = psutil.virtual_memory().total
                self.stats.allocated_memory = 0
                self.stats.peak_memory = 0
                
        except Exception as e:
            logger.error(f"Failed to initialize memory stats: {str(e)}")
            raise ResourceError(f"Failed to initialize memory stats: {str(e)}")
    
    def optimize_model_memory(self, model: torch.nn.Module) -> None:
        """Optimize model memory usage.
        
        Args:
            model: Model to optimize
        """
        try:
            if self.device == "cuda":
                # Enable gradient checkpointing if available
                if self.config.enable_gradient_checkpointing:
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                    elif hasattr(model, "enable_checkpointing"):
                        model.enable_checkpointing()
                
                # Enable memory efficient attention if available
                if self.config.enable_attention_offloading:
                    if hasattr(model, "enable_memory_efficient_attention"):
                        model.enable_memory_efficient_attention()
                
                # Enable template offloading if available
                if self.config.enable_template_offloading:
                    if hasattr(model, "enable_template_offloading"):
                        model.enable_template_offloading()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
            logger.info("Optimized model memory usage")
            
        except Exception as e:
            logger.error(f"Failed to optimize model memory: {str(e)}")
            raise ModelMemoryError(f"Model memory optimization failed: {str(e)}")
    
    def optimize_batch_memory(
        self,
        batch_size: int,
        sequence_length: int,
        num_templates: Optional[int] = None
    ) -> Tuple[int, int]:
        """Optimize batch size and chunk size based on available memory.
        
        Args:
            batch_size: Current batch size
            sequence_length: Length of sequences
            num_templates: Optional number of templates
            
        Returns:
            Tuple of (optimized_batch_size, chunk_size)
        """
        try:
            # Calculate memory requirements
            base_memory = self._estimate_memory_requirements(
                batch_size, sequence_length, num_templates
            )
            
            # Get available memory
            available_memory = self._get_available_memory()
            
            # Adjust batch size if needed
            if base_memory > available_memory:
                new_batch_size = int(batch_size * (available_memory / base_memory))
                new_batch_size = max(1, new_batch_size)
                logger.info(f"Reduced batch size from {batch_size} to {new_batch_size}")
            else:
                new_batch_size = batch_size
            
            # Calculate optimal chunk size
            chunk_size = min(
                self.config.chunk_size,
                sequence_length,
                int(available_memory / (base_memory / sequence_length))
            )
            
            return new_batch_size, chunk_size
            
        except Exception as e:
            logger.error(f"Failed to optimize batch memory: {str(e)}")
            raise ModelMemoryError(f"Batch memory optimization failed: {str(e)}")
    
    def _estimate_memory_requirements(
        self,
        batch_size: int,
        sequence_length: int,
        num_templates: Optional[int] = None
    ) -> float:
        """Estimate memory requirements for a batch.
        
        Args:
            batch_size: Batch size
            sequence_length: Length of sequences
            num_templates: Optional number of templates
            
        Returns:
            Estimated memory requirement in GB
        """
        # Base memory for sequence processing
        base_memory = batch_size * sequence_length * sequence_length * 4  # 4 bytes per float
        
        # Add template memory if applicable
        if num_templates is not None:
            template_memory = batch_size * num_templates * sequence_length * 4
            base_memory += template_memory
        
        # Add overhead for model parameters and gradients
        base_memory *= 1.5
        
        return base_memory / 1e9  # Convert to GB
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB."""
        if self.device == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return psutil.virtual_memory().available / 1e9
    
    def check_memory_available(self, required_memory: int) -> bool:
        """Check if there is enough available memory.
        
        Args:
            required_memory: Required memory in bytes
            
        Returns:
            bool: True if enough memory is available, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                return True
                
            available_memory = self.stats.total_memory - self.stats.allocated_memory
            
            if self.config.max_gpu_memory:
                available_memory = min(available_memory, int(self.config.max_gpu_memory * 1e9) - self.stats.allocated_memory)
            
            return available_memory >= required_memory
            
        except Exception as e:
            logger.error(f"Failed to check memory availability: {str(e)}")
            raise MemoryError(f"Failed to check memory availability: {str(e)}")
    
    def allocate_memory(self, size: int, description: str = "") -> bool:
        """Allocate memory.
        
        Args:
            size: Amount of memory to allocate in bytes
            description: Description of the allocation
            
        Returns:
            bool: True if allocation successful, False otherwise
        """
        try:
            if not self.check_memory_available(size):
                return False
            
            self.stats.allocated_memory += size
            current_allocated = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
            
            if current_allocated > self.stats.peak_memory:
                self.stats.peak_memory = current_allocated
                self.stats.last_peak_time = datetime.now()
            
            # Record allocation
            self.stats.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'size': size,
                'description': description,
                'total_allocated': self.stats.allocated_memory
            })
            
            # Update memory history
            self._update_memory_history()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate memory: {str(e)}")
            raise MemoryError(f"Failed to allocate memory: {str(e)}")
    
    def free_memory(self, size: int, description: str = "") -> None:
        """Free allocated memory.
        
        Args:
            size: Amount of memory to free in bytes
            description: Description of the memory being freed
        """
        try:
            self.stats.allocated_memory = max(0, self.stats.allocated_memory - size)
            
            # Record memory free
            self.stats.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'size': -size,
                'description': f"Freed: {description}",
                'total_allocated': self.stats.allocated_memory
            })
            
            # Update memory history
            self._update_memory_history()
            
        except Exception as e:
            logger.error(f"Failed to free memory: {str(e)}")
            raise MemoryError(f"Failed to free memory: {str(e)}")
    
    def clear_memory(self) -> None:
        """Clear all allocated memory."""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Reset memory tracking
            self.stats.allocated_memory = 0
            self.stats.peak_memory = 0
            
            # Record memory clear
            self.stats.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'size': -self.stats.allocated_memory,
                'description': "Cleared all memory",
                'total_allocated': 0
            })
            
            # Update memory history
            self._update_memory_history()
            
            logger.info("Cleared memory")
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {str(e)}")
            raise MemoryError(f"Memory clearing failed: {str(e)}")
    
    def _update_memory_history(self) -> None:
        """Update memory usage history."""
        try:
            current_memory = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
            self.stats.memory_history.append({
                'timestamp': datetime.now().isoformat(),
                'allocated': current_memory,
                'peak': self.stats.peak_memory,
                'total': self.stats.total_memory
            })
            
            # Keep only last 1000 entries
            if len(self.stats.memory_history) > 1000:
                self.stats.memory_history = self.stats.memory_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to update memory history: {str(e)}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary of memory usage statistics
        """
        try:
            stats = {
                "cpu_total": psutil.virtual_memory().total / 1e9,
                "cpu_available": psutil.virtual_memory().available / 1e9,
                "cpu_used": psutil.virtual_memory().used / 1e9,
                "peak_memory": self.stats.peak_memory / 1e9,
                "current_memory": self.stats.allocated_memory / 1e9
            }
            
            if self.device == "cuda":
                stats.update({
                    "gpu_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "gpu_allocated": torch.cuda.memory_allocated(0) / 1e9,
                    "gpu_cached": torch.cuda.memory_reserved(0) / 1e9
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {str(e)}")
            raise MemoryError(f"Memory usage retrieval failed: {str(e)}")
    
    def monitor_memory(self, threshold: float = 0.9) -> bool:
        """Monitor memory usage and check if it exceeds threshold.
        
        Args:
            threshold: Memory usage threshold (0-1)
            
        Returns:
            True if memory usage is below threshold
        """
        try:
            if self.device == "cuda":
                memory_usage = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
            else:
                memory_usage = psutil.virtual_memory().used / psutil.virtual_memory().total
            
            return memory_usage < threshold
            
        except Exception as e:
            logger.error(f"Failed to monitor memory: {str(e)}")
            raise MemoryError(f"Memory monitoring failed: {str(e)}")
    
    # Model weight cache methods
    def _load_cache_index(self) -> None:
        """Load the cache index from disk."""
        index_file = self.cache_dir / 'cache_index.json'
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {str(e)}")
                self.cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        try:
            index_file = self.cache_dir / 'cache_index.json'
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {str(e)}")
    
    def has_weights(self, model_name: str) -> bool:
        """Check if weights are cached for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if weights are cached, False otherwise
        """
        if model_name not in self.cache_index:
            return False
            
        weight_file = self.cache_dir / self.cache_index[model_name]
        return weight_file.exists()
    
    def get_weights(self, model_name: str) -> Dict:
        """Get cached weights for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Model weights
            
        Raises:
            KeyError: If weights not found in cache
        """
        if not self.has_weights(model_name):
            raise KeyError(f"No cached weights found for {model_name}")
            
        try:
            weight_file = self.cache_dir / self.cache_index[model_name]
            return torch.load(weight_file)
            
        except Exception as e:
            logger.error(f"Failed to load cached weights for {model_name}: {str(e)}")
            raise ResourceError(f"Failed to load cached weights: {str(e)}")
    
    def cache_weights(self, model_name: str, model: Any) -> None:
        """Cache weights for a model.
        
        Args:
            model_name: Name of the model
            model: Model instance to cache weights from
        """
        try:
            # Check cache size limit
            if self.config.max_cache_size:
                current_size = self._get_cache_size()
                if current_size > self.config.max_cache_size * 1e9:
                    self._cleanup_cache()
            
            weight_file = self.cache_dir / f"{model_name}_weights.pt"
            torch.save(model.state_dict(), weight_file)
            
            self.cache_index[model_name] = str(weight_file.name)
            self._save_cache_index()
            
            logger.info(f"Cached weights for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to cache weights for {model_name}: {str(e)}")
            raise ResourceError(f"Failed to cache weights: {str(e)}")
    
    def _get_cache_size(self) -> float:
        """Get current cache size in bytes."""
        return sum(
            (self.cache_dir / filename).stat().st_size
            for filename in self.cache_index.values()
        )
    
    def _cleanup_cache(self) -> None:
        """Clean up cache to stay within size limit."""
        try:
            # Sort files by last access time
            files = sorted(
                [(f, f.stat().st_atime) for f in self.cache_dir.glob('*_weights.pt')],
                key=lambda x: x[1]
            )
            
            # Remove oldest files until under limit
            while self._get_cache_size() > self.config.max_cache_size * 1e9 and files:
                file_path, _ = files.pop(0)
                file_path.unlink()
                
            # Update cache index
            self.cache_index = {
                name: str(f.name)
                for f in self.cache_dir.glob('*_weights.pt')
            }
            self._save_cache_index()
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {str(e)}")
            raise ResourceError(f"Cache cleanup failed: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached weights."""
        try:
            for weight_file in self.cache_dir.glob('*_weights.pt'):
                weight_file.unlink()
            
            self.cache_index = {}
            self._save_cache_index()
            
            logger.info("Cleared model weight cache")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise ResourceError(f"Failed to clear cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            total_size = self._get_cache_size()
            
            return {
                'num_models': len(self.cache_index),
                'total_size': total_size,
                'cache_dir': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {'num_models': 0, 'total_size': 0, 'cache_dir': str(self.cache_dir)}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_memory()

    def optimize_cpu_gpu_split(self):
        """Optimize workload distribution between CPU and GPU"""
        pass
        
    def dynamic_batch_sizing(self):
        """Dynamically adjust batch sizes based on memory availability"""
        pass 