"""Tests for model versioning and caching functionality."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from proteus.core.versioning import (
    ModelVersionManager,
    ModelCacheManager,
    DistributedVersionManager,
    VersionConflictError,
    InvalidVersionError,
    CacheMemoryError
)
from proteus.interfaces import ModelInterface, BaseModelConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def version_manager(temp_dir):
    """Create a version manager instance."""
    return ModelVersionManager(
        storage_path=temp_dir / "versions",
        backup_enabled=True
    )

@pytest.fixture
def cache_manager():
    """Create a cache manager instance."""
    return ModelCacheManager(
        max_memory_gb=1,
        eviction_policy="lru"
    )

@pytest.fixture
def sample_model():
    """Create a sample model instance."""
    return ModelInterface(
        BaseModelConfig(
            name="test_model",
            model_type="esm"
        )
    )

def test_version_registration(version_manager, sample_model):
    """Test model version registration."""
    # Register initial version
    version_manager.register_version(
        model=sample_model,
        version="1.0.0",
        metadata={
            "changes": "Initial version"
        }
    )
    
    # Verify version exists
    versions = version_manager.list_versions("test_model")
    assert "1.0.0" in versions
    
    # Try to register same version again
    with pytest.raises(VersionConflictError):
        version_manager.register_version(
            model=sample_model,
            version="1.0.0",
            metadata={"changes": "Duplicate"}
        )
    
    # Register new version
    version_manager.register_version(
        model=sample_model,
        version="1.1.0",
        metadata={
            "changes": "Updated version"
        }
    )
    
    # Verify both versions exist
    versions = version_manager.list_versions("test_model")
    assert len(versions) == 2
    assert "1.0.0" in versions
    assert "1.1.0" in versions

def test_version_retrieval(version_manager, sample_model):
    """Test model version retrieval."""
    # Register version
    metadata = {
        "training_data": "test_data",
        "parameters": "test_params"
    }
    version_manager.register_version(
        model=sample_model,
        version="1.0.0",
        metadata=metadata
    )
    
    # Retrieve version
    model = version_manager.get_version("test_model", "1.0.0")
    assert model.name == "test_model"
    assert model.metadata == metadata
    
    # Try to retrieve non-existent version
    with pytest.raises(InvalidVersionError):
        version_manager.get_version("test_model", "2.0.0")

def test_version_rollback(version_manager, sample_model):
    """Test version rollback functionality."""
    # Register multiple versions
    versions = ["1.0.0", "1.1.0", "1.2.0"]
    for version in versions:
        version_manager.register_version(
            model=sample_model,
            version=version,
            metadata={"version": version}
        )
    
    # Rollback to previous version
    version_manager.rollback("test_model", "1.1.0")
    
    # Verify current version
    current = version_manager.get_current_version("test_model")
    assert current.version == "1.1.0"
    
    # Verify version history
    history = version_manager.get_version_history("test_model")
    assert len(history) == 3
    assert history[-1]["version"] == "1.1.0"

def test_model_caching(cache_manager, sample_model):
    """Test model caching functionality."""
    # Cache model
    cache_manager.cache_model(sample_model)
    
    # Verify model is cached
    assert cache_manager.is_cached("test_model")
    
    # Get cached model
    cached_model = cache_manager.get_cached_model("test_model")
    assert cached_model.name == sample_model.name
    
    # Test cache statistics
    stats = cache_manager.get_cache_stats()
    assert stats["hits"] > 0
    assert "memory_usage_gb" in stats
    
    # Clear model cache
    cache_manager.clear_model_cache("test_model")
    assert not cache_manager.is_cached("test_model")

def test_cache_memory_limit(cache_manager):
    """Test cache memory limit handling."""
    # Create multiple large models
    large_models = [
        ModelInterface(BaseModelConfig(
            name=f"large_model_{i}",
            model_type="esm"
        ))
        for i in range(5)
    ]
    
    # Try to cache models until memory limit
    with pytest.raises(CacheMemoryError):
        for model in large_models:
            cache_manager.cache_model(model)

def test_distributed_version_control():
    """Test distributed version control functionality."""
    # Skip if no distributed backend available
    pytest.skip("Distributed backend not available")
    
    # Initialize distributed manager
    dist_manager = DistributedVersionManager(
        storage_backend="memory",
        cache_backend="memory"
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="dist_model",
            model_type="esm"
        )
    )
    
    # Register version
    dist_manager.register_version(
        model=model,
        version="1.0.0",
        metadata={"distributed": True}
    )
    
    # Verify version exists
    versions = dist_manager.list_versions("dist_model")
    assert "1.0.0" in versions
    
    # Get version from specific node
    model = dist_manager.get_version(
        "dist_model",
        "1.0.0",
        preferred_node="node1"
    )
    assert model.metadata["distributed"]

def test_version_backup_restore(version_manager, sample_model, temp_dir):
    """Test version backup and restore functionality."""
    # Register version
    version_manager.register_version(
        model=sample_model,
        version="1.0.0",
        metadata={"backup": "test"}
    )
    
    # Create backup
    backup_path = temp_dir / "backup.zip"
    version_manager.create_backup(backup_path)
    
    # Clear versions
    version_manager.clear_all_versions()
    
    # Restore from backup
    version_manager.restore_from_backup(backup_path)
    
    # Verify version is restored
    model = version_manager.get_version("test_model", "1.0.0")
    assert model.metadata["backup"] == "test"

def test_cache_eviction_policies(cache_manager, sample_model):
    """Test cache eviction policies."""
    # Test LRU eviction
    cache_manager.update_cache_config({
        "eviction_policy": "lru"
    })
    
    # Cache and access models
    models = [
        ModelInterface(BaseModelConfig(
            name=f"model_{i}",
            model_type="esm"
        ))
        for i in range(3)
    ]
    
    for model in models:
        cache_manager.cache_model(model)
    
    # Access first model to update LRU order
    _ = cache_manager.get_cached_model("model_0")
    
    # Force eviction
    cache_manager.evict_least_used()
    
    # Verify least recently used model was evicted
    assert not cache_manager.is_cached("model_1")
    assert cache_manager.is_cached("model_0")

def test_version_metadata_validation(version_manager, sample_model):
    """Test version metadata validation."""
    # Test valid metadata
    valid_metadata = {
        "training_data": "test",
        "parameters": 100,
        "performance": {
            "accuracy": 0.95,
            "loss": 0.05
        }
    }
    
    version_manager.register_version(
        model=sample_model,
        version="1.0.0",
        metadata=valid_metadata
    )
    
    # Test invalid metadata
    invalid_metadata = {
        "training_data": None,  # Should be string
        "parameters": "100",    # Should be number
        "performance": "good"   # Should be dict
    }
    
    with pytest.raises(ValueError):
        version_manager.register_version(
            model=sample_model,
            version="1.1.0",
            metadata=invalid_metadata
        ) 