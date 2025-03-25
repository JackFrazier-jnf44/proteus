import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import shutil
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..exceptions import MemoryError, ResourceError

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

class GPUMemoryManager:
    """
    Manages GPU memory allocation and tracking.
    
    This class provides utilities for managing GPU memory, including checking available
    memory, allocating memory, and clearing memory when needed.
    """
    
    def __init__(self, memory_limit: Optional[int] = None, device: str = "cuda"):
        """
        Initialize the GPU memory manager.
        
        Args:
            memory_limit: Optional memory limit in bytes
            device: Device to manage memory for
        """
        self.device = torch.device(device)
        self.memory_limit = memory_limit
        self.stats = MemoryStats()
        self._initialize_stats()
        
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
    
    def check_memory_available(self, required_memory: int) -> bool:
        """
        Check if there is enough available memory.
        
        Args:
            required_memory: Required memory in bytes
            
        Returns:
            bool: True if enough memory is available, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                return True
                
            available_memory = self.stats.total_memory - self.stats.allocated_memory
            
            if self.memory_limit:
                available_memory = min(available_memory, self.memory_limit - self.stats.allocated_memory)
            
            return available_memory >= required_memory
            
        except Exception as e:
            logger.error(f"Failed to check memory availability: {str(e)}")
            raise MemoryError(f"Failed to check memory availability: {str(e)}")
    
    def allocate_memory(self, size: int, description: str = "") -> bool:
        """
        Allocate GPU memory.
        
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
        """
        Free allocated GPU memory.
        
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
        """Clear all allocated GPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.stats.allocated_memory = 0
            
            # Record memory clear
            self.stats.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'size': -self.stats.allocated_memory,
                'description': "Cleared all memory",
                'total_allocated': 0
            })
            
            # Update memory history
            self._update_memory_history()
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {str(e)}")
            raise MemoryError(f"Failed to clear memory: {str(e)}")
    
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
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        return {
            'total_memory': self.stats.total_memory,
            'allocated_memory': self.stats.allocated_memory,
            'peak_memory': self.stats.peak_memory,
            'last_peak_time': self.stats.last_peak_time.isoformat(),
            'memory_limit': self.memory_limit,
            'device': str(self.device)
        }
    
    def save_memory_stats(self, output_file: str) -> None:
        """
        Save memory statistics to a file.
        
        Args:
            output_file: Path to save the statistics
        """
        try:
            stats = {
                'memory_stats': self.get_memory_stats(),
                'allocation_history': self.stats.allocation_history,
                'memory_history': self.stats.memory_history
            }
            
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save memory stats: {str(e)}")
            raise ResourceError(f"Failed to save memory stats: {str(e)}")

class ModelWeightCache:
    """
    Caches model weights for faster loading.
    
    This class provides utilities for caching and retrieving model weights,
    reducing the need to reload weights from disk.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model weight cache.
        
        Args:
            cache_dir: Optional directory for storing cached weights
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.model_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = {}
        self._load_cache_index()
    
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
        """
        Check if weights are cached for a model.
        
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
        """
        Get cached weights for a model.
        
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
        """
        Cache weights for a model.
        
        Args:
            model_name: Name of the model
            model: Model instance to cache weights from
        """
        try:
            weight_file = self.cache_dir / f"{model_name}_weights.pt"
            torch.save(model.state_dict(), weight_file)
            
            self.cache_index[model_name] = str(weight_file.name)
            self._save_cache_index()
            
            logger.info(f"Cached weights for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to cache weights for {model_name}: {str(e)}")
            raise ResourceError(f"Failed to cache weights: {str(e)}")
    
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
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            total_size = sum(
                (self.cache_dir / filename).stat().st_size
                for filename in self.cache_index.values()
            )
            
            return {
                'num_models': len(self.cache_index),
                'total_size': total_size,
                'cache_dir': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {'num_models': 0, 'total_size': 0, 'cache_dir': str(self.cache_dir)} 