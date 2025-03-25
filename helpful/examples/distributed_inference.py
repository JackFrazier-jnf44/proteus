"""
Example script demonstrating distributed inference capabilities of the Proteus framework.
"""

import os
from pathlib import Path
from src.core.distributed import (
    DistributedInferenceManager,
    DistributedConfig,
    DistributedWorker,
    WorkerConfig
)
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.logging import setup_logging

def basic_distributed_inference():
    """Demonstrate basic distributed inference functionality."""
    # Initialize distributed inference
    dist_inference = DistributedInferenceManager(
        num_nodes=4,
        gpu_per_node=2
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
    )
    
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    try:
        # Run distributed inference
        result = dist_inference.predict(
            model=model,
            sequence=sequence
        )
        
        # Print results
        print("\nBasic distributed inference results:")
        print(f"Prediction ID: {result['prediction_id']}")
        print(f"Confidence score: {result['confidence']:.3f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Nodes used: {result['nodes_used']}")
        
    except Exception as e:
        print(f"\nDistributed inference failed: {e}")

def advanced_distributed_configuration():
    """Demonstrate advanced distributed configuration."""
    # Configure distributed system
    config = DistributedConfig(
        num_nodes=4,
        gpu_per_node=2,
        memory_per_node="32GB",
        network_interface="eth0",
        scheduler="dask",
        max_jobs_per_node=4,
        network_timeout=300,
        retry_attempts=3
    )
    
    # Initialize with configuration
    dist_inference = DistributedInferenceManager(config)
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="alphafold2_ptm",
            model_type="alphafold"
        )
    )
    
    # Test sequences
    sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    try:
        # Process sequences with monitoring
        for i, sequence in enumerate(sequences, 1):
            print(f"\nProcessing sequence {i}...")
            
            result = dist_inference.predict(
                model=model,
                sequence=sequence
            )
            
            print(f"Prediction ID: {result['prediction_id']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
            # Get node statistics
            stats = dist_inference.get_system_stats()
            print("\nSystem statistics:")
            print(f"Active nodes: {stats['active_nodes']}")
            print(f"Total jobs: {stats['total_jobs']}")
            print(f"Average job time: {stats['avg_job_time']:.2f}s")
            
    except Exception as e:
        print(f"\nDistributed processing failed: {e}")

def node_management():
    """Demonstrate node management functionality."""
    # Initialize distributed inference
    dist_inference = DistributedInferenceManager(
        num_nodes=2,
        gpu_per_node=1
    )
    
    try:
        # Add new node
        node_config = WorkerConfig(
            hostname="worker3",
            gpu_ids=[0, 1],
            memory="16GB",
            network_interface="eth0"
        )
        
        node_id = dist_inference.add_node(node_config)
        print(f"\nAdded new node: {node_id}")
        
        # Get node metrics
        metrics = dist_inference.get_node_metrics(node_id)
        print("\nNode metrics:")
        print(f"CPU utilization: {metrics['cpu_util']:.1f}%")
        print(f"GPU utilization: {metrics['gpu_util']:.1f}%")
        print(f"Memory usage: {metrics['memory_usage']:.1f}GB")
        
        # Update node configuration
        dist_inference.update_node_config(
            node_id,
            {"max_jobs": 8}
        )
        print("\nUpdated node configuration")
        
        # Remove node
        dist_inference.remove_node(node_id)
        print(f"\nRemoved node: {node_id}")
        
    except Exception as e:
        print(f"\nNode management failed: {e}")

def distributed_error_handling():
    """Demonstrate distributed error handling and recovery."""
    # Configure with recovery options
    dist_inference = DistributedInferenceManager(
        num_nodes=4,
        recovery_strategy="failover",
        checkpoint_interval=10
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="rosettafold",
            model_type="rosetta"
        )
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "dist_checkpoint.pkl"
    
    try:
        # Run with recovery
        result = dist_inference.predict_with_recovery(
            model=model,
            sequence="MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
            checkpoint_file=checkpoint_file
        )
        
        print("\nDistributed inference with recovery completed successfully")
        
    except Exception as e:
        print(f"\nError during distributed inference: {e}")
        print("Recovering from checkpoint...")
        
        # Recover from checkpoint
        result = dist_inference.recover_from_checkpoint(checkpoint_file)
    
    finally:
        # Cleanup checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()

def performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    # Initialize with monitoring
    dist_inference = DistributedInferenceManager(
        num_nodes=4,
        enable_monitoring=True
    )
    
    try:
        # Get system-wide metrics
        system_metrics = dist_inference.get_system_stats()
        print("\nSystem-wide metrics:")
        print(f"Total throughput: {system_metrics['throughput']:.2f} seq/s")
        print(f"Success rate: {system_metrics['success_rate']:.1f}%")
        print(f"Average latency: {system_metrics['avg_latency']:.2f}ms")
        
        # Get per-node metrics
        for node_id in dist_inference.get_active_nodes():
            metrics = dist_inference.get_node_metrics(node_id)
            print(f"\nMetrics for node {node_id}:")
            print(f"CPU utilization: {metrics['cpu_util']:.1f}%")
            print(f"GPU utilization: {metrics['gpu_util']:.1f}%")
            print(f"Memory usage: {metrics['memory_usage']:.1f}GB")
            print(f"Network I/O: {metrics['network_io']:.2f}MB/s")
            print(f"Job queue length: {metrics['queue_length']}")
        
    except Exception as e:
        print(f"\nMetrics collection failed: {e}")

def main():
    """Run distributed inference examples."""
    # Setup logging
    setup_logging(level="INFO")
    
    print("Running distributed inference examples...")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    print("\n1. Basic Distributed Inference")
    basic_distributed_inference()
    
    print("\n2. Advanced Distributed Configuration")
    advanced_distributed_configuration()
    
    print("\n3. Node Management")
    node_management()
    
    print("\n4. Distributed Error Handling")
    distributed_error_handling()
    
    print("\n5. Performance Monitoring")
    performance_monitoring()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 