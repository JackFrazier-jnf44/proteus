import pytest
import numpy as np
import torch
from src.core.ensemble import (
    EnsemblePredictor, 
    EnsembleConfig, 
    EnsembleMethod,
    EnsembleError
)
from proteus.interfaces import ModelInterface, BaseModelConfig

class TestEnsemblePredictor:
    @pytest.fixture
    def setup_ensemble(self):
        """Setup basic ensemble configuration and test data."""
        config = EnsembleConfig(
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            weights={"model1": 0.6, "model2": 0.4},
            confidence_threshold=0.7
        )
        return EnsemblePredictor(config)
    
    @pytest.fixture
    def setup_predictions(self):
        """Create sample predictions for testing."""
        return {
            "model1": {
                "structure": np.random.rand(100, 3),
                "confidence": np.random.rand(100),
                "embeddings": np.random.rand(100, 1024)
            },
            "model2": {
                "structure": np.random.rand(100, 3),
                "confidence": np.random.rand(100),
                "embeddings": np.random.rand(100, 1024)
            }
        }

    def test_weighted_average_ensemble(self, setup_ensemble, setup_predictions):
        """Test weighted average ensemble prediction."""
        result = setup_ensemble.combine_predictions(setup_predictions)
        
        assert "structure" in result
        assert "confidence" in result
        assert "embeddings" in result
        assert result["structure"].shape == (100, 3)
        assert result["confidence"].shape == (100,)
        assert result["embeddings"].shape == (100, 1024)

    def test_confidence_weighted_ensemble(self):
        """Test confidence-weighted ensemble prediction."""
        config = EnsembleConfig(
            method=EnsembleMethod.CONFIDENCE_WEIGHTED,
            confidence_threshold=0.7
        )
        predictor = EnsemblePredictor(config)
        
        predictions = {
            "model1": {
                "structure": np.random.rand(100, 3),
                "confidence": np.ones(100) * 0.8  # High confidence
            },
            "model2": {
                "structure": np.random.rand(100, 3),
                "confidence": np.ones(100) * 0.6  # Low confidence
            }
        }
        
        result = predictor.combine_predictions(predictions)
        # Should favor model1 due to higher confidence
        assert np.allclose(result["structure"], predictions["model1"]["structure"])

    def test_voting_ensemble(self):
        """Test voting-based ensemble prediction."""
        config = EnsembleConfig(
            method=EnsembleMethod.VOTING,
            voting_threshold=0.5
        )
        predictor = EnsemblePredictor(config)
        
        # Create similar predictions for 3 models
        base_structure = np.random.rand(100, 3)
        predictions = {
            "model1": {"structure": base_structure + np.random.normal(0, 0.1, (100, 3))},
            "model2": {"structure": base_structure + np.random.normal(0, 0.1, (100, 3))},
            "model3": {"structure": np.random.rand(100, 3)}  # Different prediction
        }
        
        result = predictor.combine_predictions(predictions)
        # Result should be closer to model1/model2 consensus
        assert np.mean(np.abs(result["structure"] - base_structure)) < 0.2

    def test_dynamic_weighting(self):
        """Test dynamic weight adjustment based on prediction quality."""
        config = EnsembleConfig(
            method=EnsembleMethod.DYNAMIC_WEIGHTED,
            initial_weights={"model1": 0.5, "model2": 0.5},
            adaptation_rate=0.1
        )
        predictor = EnsemblePredictor(config)
        
        # Simulate sequence of predictions
        for _ in range(5):
            predictions = {
                "model1": {
                    "structure": np.random.rand(100, 3),
                    "confidence": np.random.uniform(0.8, 0.9, 100)  # Consistently high
                },
                "model2": {
                    "structure": np.random.rand(100, 3),
                    "confidence": np.random.uniform(0.5, 0.6, 100)  # Consistently low
                }
            }
            result = predictor.combine_predictions(predictions)
        
        # Model1 should have higher weight after adaptation
        assert predictor.current_weights["model1"] > predictor.current_weights["model2"]

    def test_invalid_weights(self):
        """Test handling of invalid ensemble weights."""
        with pytest.raises(ValueError):
            EnsembleConfig(
                method=EnsembleMethod.WEIGHTED_AVERAGE,
                weights={"model1": 0.6, "model2": 0.5}  # Sum > 1
            )
    
    def test_missing_predictions(self, setup_ensemble):
        """Test handling of missing predictions."""
        predictions = {
            "model1": {"structure": np.random.rand(100, 3)},
            # model2 missing
        }
        
        with pytest.raises(EnsembleError):
            setup_ensemble.combine_predictions(predictions)

    def test_inconsistent_shapes(self, setup_ensemble):
        """Test handling of inconsistent prediction shapes."""
        predictions = {
            "model1": {"structure": np.random.rand(100, 3)},
            "model2": {"structure": np.random.rand(50, 3)}  # Different size
        }
        
        with pytest.raises(EnsembleError):
            setup_ensemble.combine_predictions(predictions)

    def test_real_model_ensemble(self):
        """Test ensemble with actual model interfaces."""
        # Setup models
        configs = [
            BaseModelConfig(name="esm2_t33_650M_UR50D", model_type="esm"),
            BaseModelConfig(name="esm2_t36_3B_UR50D", model_type="esm")
        ]
        
        models = [ModelInterface(config) for config in configs]
        
        # Setup ensemble
        ensemble_config = EnsembleConfig(
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            weights={f"model{i}": 1/len(configs) for i in range(len(configs))}
        )
        ensemble = EnsemblePredictor(ensemble_config)
        
        # Test sequence
        sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
        
        # Get predictions
        predictions = {}
        for i, model in enumerate(models):
            predictions[f"model{i}"] = model.predict_structure(sequence)
        
        # Combine predictions
        result = ensemble.combine_predictions(predictions)
        
        assert "structure" in result
        assert "confidence" in result
        assert isinstance(result["structure"], np.ndarray)
        assert isinstance(result["confidence"], np.ndarray)

    def test_ensemble_with_gpu(self):
        """Test ensemble prediction with GPU support."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        config = EnsembleConfig(
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            weights={"model1": 0.6, "model2": 0.4},
            device="cuda:0"
        )
        predictor = EnsemblePredictor(config)
        
        predictions = {
            "model1": {
                "structure": torch.rand(100, 3, device="cuda:0"),
                "confidence": torch.rand(100, device="cuda:0")
            },
            "model2": {
                "structure": torch.rand(100, 3, device="cuda:0"),
                "confidence": torch.rand(100, device="cuda:0")
            }
        }
        
        result = predictor.combine_predictions(predictions)
        assert result["structure"].device.type == "cuda"
        assert result["confidence"].device.type == "cuda"

    def test_ensemble_serialization(self, setup_ensemble, setup_predictions):
        """Test saving and loading ensemble predictions."""
        # Get ensemble prediction
        result = setup_ensemble.combine_predictions(setup_predictions)
        
        # Save prediction
        output_file = "test_ensemble.npz"
        setup_ensemble.save_prediction(result, output_file)
        
        # Load prediction
        loaded = setup_ensemble.load_prediction(output_file)
        
        assert np.allclose(result["structure"], loaded["structure"])
        assert np.allclose(result["confidence"], loaded["confidence"])
        
        # Cleanup
        import os
        os.remove(output_file)

    def test_ensemble_visualization(self, setup_ensemble, setup_predictions):
        """Test ensemble visualization capabilities."""
        result = setup_ensemble.combine_predictions(setup_predictions)
        
        # Plot ensemble comparison
        plot_file = "ensemble_comparison.png"
        setup_ensemble.plot_comparison(
            predictions=setup_predictions,
            result=result,
            output_file=plot_file
        )
        
        assert os.path.exists(plot_file)
        os.remove(plot_file) 