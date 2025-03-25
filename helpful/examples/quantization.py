"""
Example script demonstrating quantization capabilities of the Proteus framework.
"""

import os
from pathlib import Path
from src.core.quantization import (
    QuantizationManager,
    QuantizationConfig,
    DistributedQuantizationManager
)
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.logging import setup_logging

def basic_quantization():
    """Demonstrate basic quantization functionality."""
    # Initialize quantization manager
    quant_manager = QuantizationManager(
        quantization_scheme="dynamic",
        precision="int8"
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
    )
    
    # Generate calibration sequences
    calibration_sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    # Quantize model
    quantized_model = quant_manager.quantize_model(
        model=model,
        calibration_data=calibration_sequences
    )
    
    # Print quantization results
    stats = quant_manager.get_quantization_stats()
    print("\nBasic quantization results:")
    print(f"Original model size: {stats['original_size_mb']:.2f} MB")
    print(f"Quantized model size: {stats['quantized_size_mb']:.2f} MB")
    print(f"Size reduction: {stats['size_reduction_percent']:.1f}%")
    print(f"Accuracy impact: {stats['accuracy_change']:.2f}%")

def advanced_quantization():
    """Demonstrate advanced quantization configuration."""
    # Configure quantization
    config = QuantizationConfig(
        scheme="static",
        precision="int8",
        per_channel=True,
        calibration_method="entropy",
        hardware_target="cpu",
        optimize_memory=True,
        enable_fusion=True,
        threading=True
    )
    
    # Initialize manager with config
    quant_manager = QuantizationManager(config)
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="alphafold2_ptm",
            model_type="alphafold"
        )
    )
    
    # Generate calibration data
    calibration_sequences = [
        "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(length)])
        for length in [50, 100, 150]
    ]
    
    # Quantize with monitoring
    def quantization_callback(stage, progress):
        print(f"\nQuantization {stage}: {progress:.1f}%")
    
    quantized_model = quant_manager.quantize_model(
        model=model,
        calibration_data=calibration_sequences,
        callbacks=[quantization_callback]
    )
    
    # Evaluate quantization
    eval_results = quant_manager.evaluate_quantization(
        model=quantized_model,
        test_data=calibration_sequences
    )
    
    print("\nAdvanced quantization results:")
    print(f"Model size reduction: {eval_results['size_reduction_percent']:.1f}%")
    print(f"Inference speedup: {eval_results['speedup_factor']:.2f}x")
    print(f"Memory reduction: {eval_results['memory_reduction_percent']:.1f}%")
    print(f"Accuracy retention: {eval_results['accuracy_retention']:.2f}%")

def quantization_with_fallback():
    """Demonstrate quantization with fallback strategies."""
    # Configure manager with fallback
    quant_manager = QuantizationManager(
        fallback_scheme="dynamic",
        allow_accuracy_drop=0.01
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="rosettafold",
            model_type="rosetta"
        )
    )
    
    # Generate test sequences
    test_sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    try:
        # Attempt aggressive quantization
        config = QuantizationConfig(
            scheme="static",
            precision="int4",
            per_channel=True
        )
        
        quantized_model = quant_manager.quantize_model_with_fallback(
            model=model,
            config=config,
            calibration_data=test_sequences
        )
        
        print("\nQuantization with fallback completed successfully")
        
    except Exception as e:
        print(f"\nFallback quantization error: {e}")
        print("Using dynamic quantization as fallback...")
        
        # Use fallback quantization
        quantized_model = quant_manager.quantize_model(
            model=model,
            quantization_scheme="dynamic",
            calibration_data=test_sequences
        )

def distributed_quantization():
    """Demonstrate distributed quantization."""
    try:
        # Initialize distributed quantization
        dist_quant = DistributedQuantizationManager(
            num_workers=4,
            hardware_targets=["cpu", "gpu"]
        )
        
        # Create model
        model = ModelInterface(
            BaseModelConfig(
                name="esm2_t33_650M_UR50D",
                model_type="esm"
            )
        )
        
        # Generate calibration data
        calibration_sequences = [
            "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(length)])
            for length in range(50, 251, 50)
        ]
        
        # Perform distributed quantization
        quantized_model = dist_quant.quantize_model(
            model=model,
            calibration_data=calibration_sequences,
            scheme="mixed_precision"
        )
        
        # Get distributed stats
        stats = dist_quant.get_distributed_stats()
        print("\nDistributed quantization results:")
        print(f"Total processing time: {stats['total_time']:.2f}s")
        print(f"Average worker utilization: {stats['avg_worker_util']:.1f}%")
        print(f"Data processed per worker: {stats['data_per_worker']} sequences")
        
    except ImportError:
        print("\nSkipped distributed quantization (distributed backend not available)")
    except Exception as e:
        print(f"\nError during distributed quantization: {e}")

def main():
    """Run quantization examples."""
    # Setup logging
    setup_logging(level="INFO")
    
    print("Running quantization examples...")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    print("\n1. Basic Quantization")
    basic_quantization()
    
    print("\n2. Advanced Quantization")
    advanced_quantization()
    
    print("\n3. Quantization with Fallback")
    quantization_with_fallback()
    
    print("\n4. Distributed Quantization")
    distributed_quantization()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 