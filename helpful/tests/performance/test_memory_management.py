import pytest
import torch
from proteus.core.memory import GPUMemoryManager, ModelCache
from proteus.exceptions import MemoryError

class TestMemoryManagement:
    def test_memory_tracking(self):
        """Test GPU memory tracking functionality."""
        manager = GPUMemoryManager()
        
        # Test initial state
        initial_usage = manager.get_memory_usage()
        assert isinstance(initial_usage, dict)
        assert all(isinstance(v, float) for v in initial_usage.values())
        
        # Test memory reservation
        test_size = 1024 * 1024  # 1MB
        manager.reserve_memory("test_model", test_size)
        assert manager.get_reserved_memory("test_model") == test_size
        
    def test_model_caching(self):
        """Test model weight caching system."""
        cache = ModelCache(max_size=2)  # Cache only 2 models
        
        # Add models to cache
        cache.add("model1", {"weights": torch.randn(100, 100)})
        cache.add("model2", {"weights": torch.randn(100, 100)})
        
        # Test cache hit
        assert cache.get("model1") is not None
        assert cache.get("model2") is not None
        
        # Test cache eviction
        cache.add("model3", {"weights": torch.randn(100, 100)})
        assert cache.get("model1") is None  # Should be evicted 