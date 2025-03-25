import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.core.batch import BatchProcessor, BatchConfig
from src.core.database import DatabaseManager
from src.core.distributed import DistributedInferenceManager, DistributedConfig
from src.core.ensemble import EnsemblePredictor, EnsembleConfig, EnsembleMethod
from src.core.memory import MemoryManager, MemoryConfig
from src.core.quantization import QuantizationManager, QuantizationConfig
from src.core.versioning import ModelVersionManager

class TestCore:
    """Core functionality tests."""
    
    def test_batch_processing(self, mock_model, tmp_path):
        """Test batch processing functionality."""
        config = BatchConfig(
            batch_size=2,
            max_workers=2,
            output_dir=str(tmp_path)
        )
        processor = BatchProcessor(mock_model, config)
        
        sequences = ["SEQUENCE1", "SEQUENCE2"]
        results = processor.process_batch(sequences)
        
        assert len(results) == 2
        for result in results.values():
            assert len(result) == 3  # structure, confidence, distance_matrix
            
    def test_distributed_inference(self, mock_model, tmp_path):
        """Test distributed inference functionality."""
        config = DistributedConfig(
            strategy="round_robin",
            devices=["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else ["cpu"],
            batch_size=2
        )
        manager = DistributedInferenceManager(mock_model, config)
        
        inputs = {"sequence": "TESTSEQ"}
        result = manager.predict(inputs)
        
        assert "structure" in result
        assert "confidence" in result
        
    def test_ensemble_prediction(self, mock_models, test_data):
        """Test ensemble prediction functionality."""
        config = EnsembleConfig(
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            weights={"model1": 0.6, "model2": 0.4}
        )
        predictor = EnsemblePredictor(config)
        
        predictions = {
            "model1": {"structure": test_data["structure1"]},
            "model2": {"structure": test_data["structure2"]}
        }
        result = predictor.combine_predictions(predictions)
        
        assert "structure" in result
        assert isinstance(result["structure"], np.ndarray)
        
    def test_memory_management(self, mock_model):
        """Test memory management functionality."""
        config = MemoryConfig(
            max_gpu_memory=1.0,  # 1GB
            max_cpu_memory=4.0   # 4GB
        )
        manager = MemoryManager(config)
        
        # Test memory allocation
        assert manager.allocate_memory(100_000_000, "test allocation")
        stats = manager.get_memory_usage()
        assert stats["allocated_memory"] > 0
        
        # Test memory cleanup
        manager.clear_memory()
        stats = manager.get_memory_usage()
        assert stats["allocated_memory"] == 0
        
    def test_model_quantization(self, mock_model):
        """Test model quantization functionality."""
        manager = QuantizationManager()
        config = QuantizationConfig(
            type="dynamic",
            dtype=torch.qint8
        )
        
        quantized_model = manager.quantize_model(mock_model, "test_model", config)
        assert quantized_model is not None
        
        # Verify model size reduction
        original_size = manager._get_model_size(mock_model)
        quantized_size = manager._get_model_size(quantized_model)
        assert quantized_size < original_size
        
    def test_model_versioning(self, mock_model, tmp_path):
        """Test model versioning functionality."""
        version_manager = ModelVersionManager(str(tmp_path))
        
        # Add version
        version_id = version_manager.add_version(
            model_name="test_model",
            model_type="test",
            model_path=str(tmp_path / "weights.pt"),
            metadata={"version": "1.0.0"}
        )
        
        # Verify version
        version = version_manager.get_version(version_id)
        assert version.model_name == "test_model"
        assert version.metadata["version"] == "1.0.0"
        
        # Add child version
        child_id = version_manager.add_version(
            model_name="test_model",
            model_type="test", 
            model_path=str(tmp_path / "weights_v2.pt"),
            metadata={"version": "1.1.0"},
            parent_version=version_id
        )
        
        # Verify version history
        history = version_manager.get_version_history(child_id)
        assert len(history) == 2
        assert history[0].version_id == version_id 