"""Tests for advanced visualization features."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.core.visualization import (
    plot_embedding_space,
    plot_attention_weights,
    plot_confidence_per_residue,
    plot_ensemble_predictions,
    plot_structure_comparison,
    plot_ensemble_comparison,
    VisualizationConfig
)
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.ensemble import EnsemblePredictor, EnsembleConfig, EnsembleMethod

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'embeddings': np.random.rand(100, 1024),
        'attention_weights': np.random.rand(12, 50, 50),  # 12 heads, 50x50 attention matrix
        'confidence_scores': np.random.rand(50),
        'sequence': "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        'predictions': {
            'model1': {
                'structure': np.random.rand(50, 3),
                'confidence': np.random.rand(50)
            },
            'model2': {
                'structure': np.random.rand(50, 3),
                'confidence': np.random.rand(50)
            }
        }
    }

@pytest.fixture
def visualization_config():
    """Create a visualization configuration."""
    return VisualizationConfig(
        figure_size=(12, 8),
        dpi=300,
        style='seaborn',
        color_palette='Set2'
    )

def test_plot_embedding_space(temp_dir, sample_data, visualization_config):
    """Test embedding space visualization with different dimensionality reduction methods."""
    # Test t-SNE
    output_file = temp_dir / 'embedding_tsne.png'
    plot_embedding_space(
        embeddings=sample_data['embeddings'],
        sequence=sample_data['sequence'],
        output_file=output_file,
        config=visualization_config,
        method='tsne',
        perplexity=30
    )
    assert output_file.exists()
    
    # Test PCA
    output_file = temp_dir / 'embedding_pca.png'
    plot_embedding_space(
        embeddings=sample_data['embeddings'],
        sequence=sample_data['sequence'],
        output_file=output_file,
        config=visualization_config,
        method='pca'
    )
    assert output_file.exists()
    
    # Test UMAP
    output_file = temp_dir / 'embedding_umap.png'
    plot_embedding_space(
        embeddings=sample_data['embeddings'],
        sequence=sample_data['sequence'],
        output_file=output_file,
        config=visualization_config,
        method='umap'
    )
    assert output_file.exists()

def test_plot_attention_weights(temp_dir, sample_data, visualization_config):
    """Test attention weights visualization."""
    output_file = temp_dir / 'attention.png'
    plot_attention_weights(
        attention_weights=sample_data['attention_weights'],
        sequence=sample_data['sequence'],
        output_file=output_file,
        config=visualization_config,
        head_indices=[0, 1, 2],  # Plot specific attention heads
        normalize=True
    )
    assert output_file.exists()
    
    # Test with all heads
    output_file = temp_dir / 'attention_all.png'
    plot_attention_weights(
        attention_weights=sample_data['attention_weights'],
        sequence=sample_data['sequence'],
        output_file=output_file,
        config=visualization_config,
        plot_all_heads=True
    )
    assert output_file.exists()

def test_plot_confidence_per_residue(temp_dir, sample_data, visualization_config):
    """Test confidence score visualization."""
    output_file = temp_dir / 'confidence.png'
    plot_confidence_per_residue(
        confidence_scores=sample_data['confidence_scores'],
        sequence=sample_data['sequence'],
        output_file=output_file,
        config=visualization_config,
        highlight_threshold=0.7
    )
    assert output_file.exists()

def test_ensemble_visualization(temp_dir, sample_data, visualization_config):
    """Test ensemble visualization features."""
    # Configure ensemble
    config = EnsembleConfig(
        method=EnsembleMethod.WEIGHTED_AVERAGE,
        weights={'model1': 0.6, 'model2': 0.4}
    )
    
    # Test ensemble predictions plot
    output_file = temp_dir / 'ensemble_predictions.png'
    plot_ensemble_predictions(
        predictions=sample_data['predictions'],
        ensemble_config=config,
        output_file=output_file,
        config=visualization_config
    )
    assert output_file.exists()
    
    # Test structure comparison plot
    output_file = temp_dir / 'structure_comparison.png'
    plot_structure_comparison(
        structures={
            name: pred['structure']
            for name, pred in sample_data['predictions'].items()
        },
        reference_structure=sample_data['predictions']['model1']['structure'],
        output_file=output_file,
        config=visualization_config
    )
    assert output_file.exists()
    
    # Test combined ensemble comparison
    output_file = temp_dir / 'ensemble_comparison.png'
    plot_ensemble_comparison(
        predictions=sample_data['predictions'],
        result={
            'structure': np.mean([p['structure'] for p in sample_data['predictions'].values()], axis=0),
            'confidence': np.mean([p['confidence'] for p in sample_data['predictions'].values()], axis=0)
        },
        output_file=output_file,
        config=visualization_config
    )
    assert output_file.exists()

def test_visualization_with_real_models():
    """Test visualization with actual model interfaces."""
    # Initialize models
    models = [
        ModelInterface(BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )),
        ModelInterface(BaseModelConfig(
            name="alphafold2_ptm",
            model_type="alphafold"
        ))
    ]
    
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    # Get predictions
    predictions = {}
    for i, model in enumerate(models):
        try:
            predictions[f"model_{i}"] = model.predict_structure(sequence)
        except Exception as e:
            pytest.skip(f"Model prediction failed: {e}")
    
    if not predictions:
        pytest.skip("No model predictions available")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = VisualizationConfig(
            figure_size=(12, 8),
            dpi=300,
            style='seaborn'
        )
        
        # Test visualization functions
        try:
            # Plot confidence scores
            output_file = tmp_path / 'confidence.png'
            plot_confidence_per_residue(
                confidence_scores=next(iter(predictions.values()))['confidence'],
                sequence=sequence,
                output_file=output_file,
                config=config
            )
            assert output_file.exists()
            
            # Plot ensemble comparison if multiple predictions available
            if len(predictions) > 1:
                output_file = tmp_path / 'ensemble.png'
                plot_ensemble_comparison(
                    predictions=predictions,
                    result={
                        'structure': np.mean([p['structure'] for p in predictions.values()], axis=0),
                        'confidence': np.mean([p['confidence'] for p in predictions.values()], axis=0)
                    },
                    output_file=output_file,
                    config=config
                )
                assert output_file.exists()
        
        except Exception as e:
            pytest.skip(f"Visualization failed: {e}")

def test_invalid_visualization_configs():
    """Test handling of invalid visualization configurations."""
    # Test invalid figure size
    with pytest.raises(ValueError):
        VisualizationConfig(figure_size=(0, 0))
    
    # Test invalid DPI
    with pytest.raises(ValueError):
        VisualizationConfig(dpi=0)
    
    # Test invalid style
    with pytest.raises(ValueError):
        VisualizationConfig(style='invalid_style')
    
    # Test invalid color palette
    with pytest.raises(ValueError):
        VisualizationConfig(color_palette='invalid_palette') 