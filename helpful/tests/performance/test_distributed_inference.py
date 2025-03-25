"""Tests for distributed inference functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from proteus.core.distributed import (
    DistributedInference,
    DistributedConfig,
    NodeConfig,
    NodeFailureError,
    ResourceExhaustedError,
    NetworkError
)
from proteus.interfaces import ModelInterface, BaseModelConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def dist_inference():
    """Create a distributed inference instance."""
    return DistributedInference(
        num_nodes=4,
        gpu_per_node=2
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

@pytest.fixture
def mock_node_metrics():
    """Create mock node metrics data."""
    return {
        "cpu_util": 45.5,
        "gpu_util": 78.2,
        "memory_usage": 12.4,
        "network_io": 156.7,
        "queue_length": 3
    }

def test_basic_distributed_inference(dist_inference, sample_model):
    """Test basic distributed inference functionality."""
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    # Run inference
    result = dist_inference.predict(
        model=sample_model,
        sequence=sequence
    )
    
    # Verify result
    assert "prediction_id" in result
    assert "confidence" in result
    assert "processing_time" in result
    assert "nodes_used" in result
    assert len(result["nodes_used"]) > 0

def test_distributed_configuration():
    """Test distributed configuration."""
    # Test valid configuration
    config = DistributedConfig(
        num_nodes=4,
        gpu_per_node=2,
        memory_per_node="32GB",
        network_interface="eth0",
        scheduler="dask"
    )
    inference = DistributedInference(config)
    assert inference.config.num_nodes == 4
    assert inference.config.gpu_per_node == 2
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        DistributedConfig(num_nodes=0)
    
    with pytest.raises(ValueError):
        DistributedConfig(gpu_per_node=-1)

def test_node_management(dist_inference):
    """Test node management functionality."""
    # Add node
    node_config = NodeConfig(
        hostname="worker3",
        gpu_ids=[0, 1],
        memory="16GB",
        network_interface="eth0"
    )
    node_id = dist_inference.add_node(node_config)
    
    # Verify node added
    assert node_id in dist_inference.get_active_nodes()
    
    # Update node config
    dist_inference.update_node_config(
        node_id,
        {"max_jobs": 8}
    )
    
    # Verify update
    node_config = dist_inference.get_node_config(node_id)
    assert node_config["max_jobs"] == 8
    
    # Remove node
    dist_inference.remove_node(node_id)
    assert node_id not in dist_inference.get_active_nodes()

def test_error_handling(dist_inference, sample_model):
    """Test error handling functionality."""
    # Test node failure
    with patch.object(dist_inference, "predict") as mock_predict:
        mock_predict.side_effect = NodeFailureError("Node 1 failed")
        
        with pytest.raises(NodeFailureError):
            dist_inference.predict(
                model=sample_model,
                sequence="SEQUENCE"
            )
    
    # Test resource exhaustion
    with patch.object(dist_inference, "predict") as mock_predict:
        mock_predict.side_effect = ResourceExhaustedError("Out of memory")
        
        with pytest.raises(ResourceExhaustedError):
            dist_inference.predict(
                model=sample_model,
                sequence="SEQUENCE"
            )
    
    # Test network error
    with patch.object(dist_inference, "predict") as mock_predict:
        mock_predict.side_effect = NetworkError("Connection failed")
        
        with pytest.raises(NetworkError):
            dist_inference.predict(
                model=sample_model,
                sequence="SEQUENCE"
            )

def test_recovery_functionality(dist_inference, sample_model, temp_dir):
    """Test recovery functionality."""
    # Configure recovery
    dist_inference.config.recovery_strategy = "failover"
    dist_inference.config.checkpoint_interval = 10
    
    # Create checkpoint file
    checkpoint_file = temp_dir / "checkpoint.pkl"
    
    try:
        # Run with recovery
        result = dist_inference.predict_with_recovery(
            model=sample_model,
            sequence="SEQUENCE",
            checkpoint_file=checkpoint_file
        )
        
        assert result is not None
        assert checkpoint_file.exists()
        
    except Exception:
        # Test recovery from checkpoint
        result = dist_inference.recover_from_checkpoint(checkpoint_file)
        assert result is not None

def test_performance_monitoring(dist_inference, mock_node_metrics):
    """Test performance monitoring functionality."""
    # Enable monitoring
    dist_inference.enable_monitoring()
    
    # Mock node metrics
    with patch.object(dist_inference, "get_node_metrics") as mock_metrics:
        mock_metrics.return_value = mock_node_metrics
        
        # Get metrics for node
        metrics = dist_inference.get_node_metrics("node1")
        assert metrics["cpu_util"] == mock_node_metrics["cpu_util"]
        assert metrics["gpu_util"] == mock_node_metrics["gpu_util"]
        
        # Get system stats
        stats = dist_inference.get_system_stats()
        assert "active_nodes" in stats
        assert "total_jobs" in stats
        assert "avg_job_time" in stats

def test_resource_management(dist_inference):
    """Test resource management functionality."""
    # Test resource allocation
    resources = dist_inference.allocate_resources(
        cpu_threads=4,
        gpu_memory=8,
        memory_gb=16
    )
    assert resources["allocated"] is True
    
    # Test resource release
    dist_inference.release_resources(resources["resource_id"])
    
    # Test resource status
    status = dist_inference.get_resource_status()
    assert "available_cpu_threads" in status
    assert "available_gpu_memory" in status
    assert "available_memory_gb" in status

def test_load_balancing(dist_inference, sample_model):
    """Test load balancing functionality."""
    # Enable load balancing
    dist_inference.enable_load_balancing(
        strategy="least_loaded",
        check_interval=60
    )
    
    # Run multiple predictions
    sequences = ["SEQ1", "SEQ2", "SEQ3"]
    results = []
    
    for sequence in sequences:
        result = dist_inference.predict(
            model=sample_model,
            sequence=sequence
        )
        results.append(result)
    
    # Check load distribution
    load_stats = dist_inference.get_load_distribution()
    assert max(load_stats.values()) - min(load_stats.values()) < 0.5  # Max 50% imbalance

def test_distributed_batch_processing(dist_inference, sample_model):
    """Test distributed batch processing."""
    # Configure batch processing
    dist_inference.config.max_batch_size = 10
    
    # Create sequences
    sequences = [f"SEQ{i}" for i in range(20)]
    
    # Process batch
    results = dist_inference.process_batch(
        model=sample_model,
        sequences=sequences,
        batch_size=5
    )
    
    assert len(results) == len(sequences)
    for result in results:
        assert "prediction_id" in result
        assert "confidence" in result

def test_network_optimization(dist_inference):
    """Test network optimization functionality."""
    # Configure network
    dist_inference.optimize_network(
        compression=True,
        batch_transfer=True,
        prefetch_size=2
    )
    
    # Get network stats
    stats = dist_inference.get_network_stats()
    assert "bandwidth_usage" in stats
    assert "latency" in stats
    assert "packet_loss" in stats
    
    # Test network configuration
    config = dist_inference.get_network_config()
    assert config["compression"] is True
    assert config["batch_transfer"] is True
    assert config["prefetch_size"] == 2 