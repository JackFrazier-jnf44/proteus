"""
Example script demonstrating model versioning and caching capabilities of the Proteus framework.
"""

from src.core.versioning import (
    ModelVersionManager,
    ModelCacheManager,
    DistributedVersionManager
)
from src.interfaces import ModelInterface, BaseModelConfig
import tempfile
from pathlib import Path

def basic_version_control():
    """Demonstrate basic version control functionality."""
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
    )
    
    # Register initial version
    version_manager.register_version(
        model=model,
        version="1.0.0",
        metadata={
            "training_data": "UR50D",
            "architecture": "transformer",
            "parameters": "650M",
            "changes": "Initial release"
        }
    )
    
    # Update model and register new version
    model.update_weights("path/to/new/weights")
    version_manager.register_version(
        model=model,
        version="1.1.0",
        metadata={
            "training_data": "UR50D",
            "architecture": "transformer",
            "parameters": "650M",
            "changes": "Updated model weights"
        }
    )
    
    # List all versions
    versions = version_manager.list_versions("esm2_t33_650M_UR50D")
    print("\nAvailable versions:")
    for version in versions:
        print(f"- {version}")
    
    # Get specific version
    model_v1 = version_manager.get_version("esm2_t33_650M_UR50D", "1.0.0")
    print(f"\nLoaded version 1.0.0: {model_v1.metadata}")

def model_caching():
    """Demonstrate model caching functionality."""
    # Initialize cache manager
    cache_manager = ModelCacheManager(
        max_memory_gb=4,
        eviction_policy="lru"
    )
    
    # Create models
    models = [
        ModelInterface(BaseModelConfig(
            name=f"model_{i}",
            model_type="esm"
        ))
        for i in range(3)
    ]
    
    # Cache models
    for model in models:
        cache_manager.cache_model(model)
        print(f"\nCached {model.name}")
    
    # Get cache statistics
    stats = cache_manager.get_cache_stats()
    print("\nCache statistics:")
    print(f"- Memory usage: {stats['memory_usage_gb']:.2f} GB")
    print(f"- Cache hits: {stats['hits']}")
    print(f"- Cache misses: {stats['misses']}")
    
    # Get cached model
    cached_model = cache_manager.get_cached_model("model_0")
    print(f"\nRetrieved model from cache: {cached_model.name}")
    
    # Clear specific model from cache
    cache_manager.clear_model_cache("model_1")
    print("\nCleared model_1 from cache")
    
    # Update cache configuration
    cache_manager.update_cache_config({
        "max_memory_gb": 8,
        "eviction_policy": "mru"
    })
    print("\nUpdated cache configuration")

def distributed_version_control():
    """Demonstrate distributed version control functionality."""
    # Initialize distributed version manager
    dist_manager = DistributedVersionManager(
        storage_backend="s3",
        cache_backend="redis",
        config={
            "s3_bucket": "proteus-models",
            "redis_host": "localhost",
            "redis_port": 6379
        }
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="alphafold2_ptm",
            model_type="alphafold"
        )
    )
    
    # Register version in distributed storage
    dist_manager.register_version(
        model=model,
        version="2.0.0",
        metadata={
            "training_data": "PDB70",
            "architecture": "transformer",
            "parameters": "93M",
            "changes": "Initial distributed version"
        }
    )
    
    # List versions across all nodes
    versions = dist_manager.list_versions("alphafold2_ptm")
    print("\nAvailable versions in distributed storage:")
    for version in versions:
        print(f"- {version}")
    
    # Get version from any available node
    model_v2 = dist_manager.get_version(
        "alphafold2_ptm",
        "2.0.0",
        preferred_node="node1"
    )
    print(f"\nLoaded version 2.0.0 from distributed storage: {model_v2.metadata}")

def version_rollback():
    """Demonstrate version rollback functionality."""
    # Initialize version manager with backup
    version_manager = ModelVersionManager(backup_enabled=True)
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="rosettafold",
            model_type="rosetta"
        )
    )
    
    # Register multiple versions
    versions = ["1.0.0", "1.1.0", "1.2.0"]
    for version in versions:
        version_manager.register_version(
            model=model,
            version=version,
            metadata={
                "version": version,
                "changes": f"Version {version} changes"
            }
        )
    
    # Rollback to specific version
    version_manager.rollback("rosettafold", "1.1.0")
    print("\nRolled back to version 1.1.0")
    
    # Verify current version
    current = version_manager.get_current_version("rosettafold")
    print(f"Current version: {current.version}")
    
    # List version history
    history = version_manager.get_version_history("rosettafold")
    print("\nVersion history:")
    for entry in history:
        print(f"- {entry['version']}: {entry['timestamp']}")

def main():
    """Run model versioning examples."""
    print("Running model versioning examples...")
    
    print("\n1. Basic Version Control")
    basic_version_control()
    
    print("\n2. Model Caching")
    model_caching()
    
    print("\n3. Distributed Version Control")
    try:
        distributed_version_control()
    except Exception as e:
        print(f"Skipped distributed example: {e}")
    
    print("\n4. Version Rollback")
    version_rollback()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 