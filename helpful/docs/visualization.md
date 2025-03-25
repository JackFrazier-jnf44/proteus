# Visualization Capabilities

This document describes the visualization capabilities of the multi-model protein structure analysis framework.

## Overview

The framework provides comprehensive visualization tools for analyzing protein structure predictions and model outputs. These visualizations are organized into several categories:

1. Basic Structure Visualization
2. Ensemble Visualization
3. Structure Comparison
4. Combined Visualizations

## Basic Structure Visualization

### Contact Maps

```python
from src.core.visualization import plot_contact_map

# Plot contact map
plot_contact_map(
    distance_matrix=model.get_distance_matrix(),
    sequence="MVKVGVNG...",
    output_file="contact_map.png"
)
```

### Confidence Analysis

```python
from src.core.visualization import plot_confidence_per_residue

# Plot confidence scores
plot_confidence_per_residue(
    confidence_scores=model.get_confidence_scores(),
    sequence="MVKVGVNG...",
    output_file="confidence.png"
)
```

### Additional Basic Visualizations

- `plot_rmsd_distribution`: RMSD distribution between model predictions
- `plot_secondary_structure`: Secondary structure assignments
- `plot_attention_weights`: Attention patterns from transformer models
- `plot_embedding_space`: t-SNE visualization of model embeddings
- `plot_sequence_alignment`: Sequence alignment visualization
- `plot_evolutionary_profile`: Evolutionary conservation profile
- `plot_structural_features`: Structural feature analysis
- `plot_model_comparison`: Model comparison plots
- `plot_version_comparison`: Version comparison analysis

## Ensemble Visualization

The framework provides tools for visualizing ensemble predictions and their combinations:

```python
from src.core.visualization import (
    plot_ensemble_predictions,
    plot_structure_comparison,
    plot_ensemble_comparison
)
from src.core.ensemble import EnsembleConfig, EnsembleMethod

# Configure ensemble
ensemble_config = EnsembleConfig(
    method=EnsembleMethod.WEIGHTED_AVERAGE,
    weights={
        "model1": 0.4,
        "model2": 0.3,
        "model3": 0.3
    }
)

# Plot ensemble predictions
plot_ensemble_predictions(
    predictions={
        "model1": model1_predictions,
        "model2": model2_predictions,
        "model3": model3_predictions
    },
    ensemble_config=ensemble_config,
    output_file="ensemble_predictions.png"
)
```

## Structure Comparison

Visualize structural comparisons between different models and reference structures:

```python
# Plot structure comparison metrics
plot_structure_comparison(
    structures={
        "model1": "model1.pdb",
        "model2": "model2.pdb",
        "model3": "model3.pdb"
    },
    reference_structure="reference.pdb",
    output_file="structure_comparison.png"
)
```

## Combined Visualizations

Combine ensemble predictions with structure comparison metrics:

```python
# Plot combined ensemble and comparison visualization
plot_ensemble_comparison(
    predictions={
        "model1": model1_predictions,
        "model2": model2_predictions,
        "model3": model3_predictions
    },
    ensemble_config=ensemble_config,
    structures={
        "model1": "model1.pdb",
        "model2": "model2.pdb",
        "model3": "model3.pdb"
    },
    reference_structure="reference.pdb",
    output_file="ensemble_comparison.png"
)
```

## Visualization Configuration

Visualizations can be customized using the `VisualizationConfig` class:

```python
from src.core.visualization import VisualizationConfig

config = VisualizationConfig(
    figure_size=(15, 10),
    dpi=300,
    style='seaborn',
    color_palette='Set2',
    output_format='png',
    call_model=True,  # Enable model-specific visualizations
    use_ensemble=True,  # Enable ensemble visualizations
    use_comparison=True  # Enable structure comparison visualizations
)

# Use configuration in visualization calls
plot_ensemble_predictions(
    predictions=predictions,
    ensemble_config=ensemble_config,
    output_file="ensemble.png",
    config=config
)
```

## Available Visualization Types

1. **Basic Structure Visualization**
   - Contact maps
   - Confidence scores per residue
   - RMSD distribution
   - Secondary structure
   - Attention weights
   - Embedding space
   - Sequence alignment
   - Evolutionary profile
   - Structural features
   - Model comparison
   - Version comparison

2. **Ensemble Visualization**
   - Individual model predictions
   - Ensemble predictions
   - Confidence scores
   - Model weights

3. **Structure Comparison**
   - RMSD
   - TM-Score
   - Contact map overlap
   - Secondary structure similarity

4. **Combined Visualizations**
   - Ensemble predictions with structure comparison
   - Model comparison with confidence scores
   - Version comparison with structural metrics

## Dependencies

The visualization module requires the following Python packages:

- matplotlib
- seaborn
- numpy
- scikit-learn
- MDAnalysis
- BioPython

## Notes

1. Some visualizations may not be available depending on the model outputs:
   - Attention weights are only available for transformer-based models
   - Confidence scores may vary by model type
   - Secondary structure requires valid PDB files

2. For large proteins or many models:
   - t-SNE visualization may be slow
   - Memory usage may be high for embedding comparisons
   - Consider downsampling or using PCA for large datasets

3. The framework automatically handles:
   - GPU acceleration when available
   - Memory management for large datasets
   - Error handling and logging
   - Consistent styling across plots
