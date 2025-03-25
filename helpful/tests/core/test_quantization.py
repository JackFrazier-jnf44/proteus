"""Tests for quantization functionality."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from proteus.core.quantization import (
    QuantizationManager,
    QuantizationConfig,
    DistributedQuantizationManager,
    QuantizationError,
    CalibrationError
)
from proteus.interfaces import ModelInterface, BaseModelConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def quant_manager():
    """Create a quantization manager instance."""
    return QuantizationManager(
        quantization_scheme="dynamic",
        precision="int8"
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
def calibration_data():
    """Create sample calibration data."""
    return [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]

def test_basic_quantization(quant_manager, sample_model, calibration_data):
    """Test basic quantization functionality."""
    # Quantize model
    quantized_model = quant_manager.quantize_model(
        model=sample_model,
        calibration_data=calibration_data
    )
    
    # Check quantization results
    stats = quant_manager.get_quantization_stats()
    assert stats['original_size_mb'] > stats['quantized_size_mb']
    assert stats['size_reduction_percent'] > 0
    assert abs(stats['accuracy_change']) < 5.0  # Allow small accuracy change

def test_quantization_config():
    """Test quantization configuration."""
    # Test valid configuration
    config = QuantizationConfig(
        scheme="static",
        precision="int8",
        per_channel=True,
        calibration_method="entropy"
    )
    manager = QuantizationManager(config)
    assert manager.config.scheme == "static"
    assert manager.config.precision == "int8"
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        QuantizationConfig(scheme="invalid")
    
    with pytest.raises(ValueError):
        QuantizationConfig(precision="int3")

def test_quantization_callbacks(quant_manager, sample_model, calibration_data):
    """Test quantization callbacks."""
    # Initialize callback tracking
    callback_data = {
        'stages_completed': 0,
        'total_progress': 0
    }
    
    def quant_callback(stage, progress):
        callback_data['stages_completed'] += 1
        callback_data['total_progress'] = progress
    
    # Quantize with callback
    quantized_model = quant_manager.quantize_model(
        model=sample_model,
        calibration_data=calibration_data,
        callbacks=[quant_callback]
    )
    
    # Check callback data
    assert callback_data['stages_completed'] > 0
    assert callback_data['total_progress'] == 100.0

def test_quantization_error_handling(quant_manager, sample_model):
    """Test quantization error handling."""
    # Test with invalid calibration data
    invalid_data = ["INVALID", "NOT_AA_SEQUENCE", "123"]
    
    with pytest.raises(CalibrationError):
        quant_manager.quantize_model(
            model=sample_model,
            calibration_data=invalid_data
        )
    
    # Test with incompatible configuration
    with pytest.raises(QuantizationError):
        quant_manager.quantize_model(
            model=sample_model,
            calibration_data=[],
            precision="int4"  # Unsupported precision
        )

def test_quantization_with_fallback(quant_manager, sample_model, calibration_data):
    """Test quantization fallback functionality."""
    # Configure fallback
    manager = QuantizationManager(
        fallback_scheme="dynamic",
        allow_accuracy_drop=0.01
    )
    
    # Test aggressive quantization with fallback
    config = QuantizationConfig(
        scheme="static",
        precision="int4",
        per_channel=True
    )
    
    try:
        quantized_model = manager.quantize_model_with_fallback(
            model=sample_model,
            config=config,
            calibration_data=calibration_data
        )
    except QuantizationError:
        # Should fall back to dynamic quantization
        quantized_model = manager.quantize_model(
            model=sample_model,
            quantization_scheme="dynamic",
            calibration_data=calibration_data
        )
    
    assert quantized_model is not None

def test_distributed_quantization():
    """Test distributed quantization functionality."""
    try:
        # Initialize distributed manager
        dist_quant = DistributedQuantizationManager(
            num_workers=2,
            hardware_targets=["cpu"]
        )
        
        # Create model and data
        model = ModelInterface(
            BaseModelConfig(
                name="test_model",
                model_type="esm"
            )
        )
        
        calibration_data = [
            "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(50)])
            for _ in range(3)
        ]
        
        # Perform distributed quantization
        quantized_model = dist_quant.quantize_model(
            model=model,
            calibration_data=calibration_data,
            scheme="mixed_precision"
        )
        
        # Check distributed stats
        stats = dist_quant.get_distributed_stats()
        assert stats['total_time'] > 0
        assert 0 <= stats['avg_worker_util'] <= 100
        assert stats['data_per_worker'] > 0
        
    except ImportError:
        pytest.skip("Distributed backend not available")

def test_hardware_specific_quantization(quant_manager, sample_model, calibration_data):
    """Test hardware-specific quantization."""
    # Configure for specific hardware
    config = QuantizationConfig(
        hardware_target="cpu",
        optimize_memory=True,
        enable_fusion=True,
        threading=True
    )
    
    # Quantize for specific hardware
    quantized_model = quant_manager.quantize_model(
        model=sample_model,
        config=config,
        calibration_data=calibration_data
    )
    
    # Verify hardware-specific optimizations
    assert quantized_model.config.hardware_target == "cpu"
    assert quantized_model.config.enable_fusion is True

def test_quantization_export_import(quant_manager, sample_model, calibration_data, temp_dir):
    """Test quantization model export and import."""
    # Quantize model
    quantized_model = quant_manager.quantize_model(
        model=sample_model,
        calibration_data=calibration_data
    )
    
    # Export model
    export_path = temp_dir / "quantized_model.pt"
    quant_manager.export_quantized_model(
        model=quantized_model,
        path=export_path
    )
    
    # Import model
    imported_model = quant_manager.import_quantized_model(
        path=export_path
    )
    
    # Verify imported model
    assert imported_model.config.precision == quantized_model.config.precision
    assert imported_model.config.scheme == quantized_model.config.scheme

def test_calibration_methods(quant_manager, sample_model, calibration_data):
    """Test different calibration methods."""
    calibration_methods = ["minmax", "entropy", "percentile"]
    
    for method in calibration_methods:
        # Configure calibration
        config = QuantizationConfig(
            calibration_method=method
        )
        
        # Quantize with specific calibration
        quantized_model = quant_manager.quantize_model(
            model=sample_model,
            config=config,
            calibration_data=calibration_data
        )
        
        # Verify calibration
        calibration_stats = quant_manager.get_calibration_stats()
        assert calibration_stats['method'] == method
        assert calibration_stats['success'] is True 