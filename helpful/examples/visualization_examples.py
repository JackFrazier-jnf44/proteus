"""
Example script demonstrating various visualization capabilities of the Proteus framework.
"""

import numpy as np
from src.core.visualization import (
    plot_contact_map,
    plot_confidence_per_residue,
    plot_rmsd_distribution,
    plot_secondary_structure,
    plot_attention_weights,
    plot_embedding_space,
    plot_sequence_alignment,
    plot_evolutionary_profile,
    plot_structural_features,
    plot_model_comparison,
    plot_version_comparison,
    VisualizationConfig
)
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.ensemble import EnsemblePredictor, EnsembleConfig, EnsembleMethod

def basic_structure_visualization(model, sequence):
    """Demonstrate basic structure visualization capabilities."""
    # Configure visualization
    config = VisualizationConfig(
        figure_size=(12, 8),
        dpi=300,
        style='seaborn',
        color_palette='Set2'
    )
    
    # 1. Contact Map
    distance_matrix = model.get_distance_matrix(sequence)
    plot_contact_map(
        distance_matrix=distance_matrix,
        sequence=sequence,
        output_file="outputs/contact_map.png",
        config=config
    )
    
    # 2. Confidence Analysis
    confidence_scores = model.get_confidence_scores(sequence)
    plot_confidence_per_residue(
        confidence_scores=confidence_scores,
        sequence=sequence,
        output_file="outputs/confidence.png",
        config=config
    )
    
    # 3. Secondary Structure
    structure = model.predict_structure(sequence)
    plot_secondary_structure(
        structure=structure,
        sequence=sequence,
        output_file="outputs/secondary_structure.png",
        config=config
    )
    
    # 4. Attention Weights (for transformer-based models)
    if hasattr(model, 'get_attention_weights'):
        attention_weights = model.get_attention_weights(sequence)
        plot_attention_weights(
            attention_weights=attention_weights,
            sequence=sequence,
            output_file="outputs/attention.png",
            config=config
        )
    
    print("Basic structure visualizations saved in outputs/")

def ensemble_visualization(models, sequence):
    """Demonstrate ensemble visualization capabilities."""
    # Configure ensemble
    config = EnsembleConfig(
        method=EnsembleMethod.WEIGHTED_AVERAGE,
        weights={f"model_{i}": 1/len(models) for i in range(len(models))}
    )
    ensemble = EnsemblePredictor(config)
    
    # Get predictions from all models
    predictions = {
        f"model_{i}": model.predict_structure(sequence)
        for i, model in enumerate(models)
    }
    
    # Combine predictions
    result = ensemble.combine_predictions(predictions)
    
    # Visualization config
    vis_config = VisualizationConfig(
        figure_size=(15, 10),
        dpi=300,
        style='seaborn',
        color_palette='Set2',
        use_ensemble=True
    )
    
    # 1. Plot individual model predictions
    plot_ensemble_predictions(
        predictions=predictions,
        ensemble_config=config,
        output_file="outputs/ensemble_predictions.png",
        config=vis_config
    )
    
    # 2. Plot structure comparison
    plot_structure_comparison(
        structures={
            f"model_{i}": pred["structure"]
            for i, pred in predictions.items()
        },
        reference_structure=result["structure"],
        output_file="outputs/structure_comparison.png",
        config=vis_config
    )
    
    # 3. Plot combined visualization
    plot_ensemble_comparison(
        predictions=predictions,
        result=result,
        output_file="outputs/ensemble_comparison.png",
        config=vis_config
    )
    
    print("Ensemble visualizations saved in outputs/")

def advanced_visualization(model, sequence):
    """Demonstrate advanced visualization capabilities."""
    # Get model embeddings
    embeddings = model.get_embeddings(sequence)
    
    # Configure visualization
    config = VisualizationConfig(
        figure_size=(12, 8),
        dpi=300,
        style='seaborn',
        color_palette='Set2'
    )
    
    # 1. Embedding Space Visualization
    plot_embedding_space(
        embeddings=embeddings,
        sequence=sequence,
        output_file="outputs/embedding_space.png",
        config=config,
        method='tsne',  # or 'pca', 'umap'
        perplexity=30
    )
    
    # 2. RMSD Distribution
    structures = [
        model.predict_structure(sequence)
        for _ in range(5)  # Multiple predictions
    ]
    plot_rmsd_distribution(
        structures=structures,
        output_file="outputs/rmsd_distribution.png",
        config=config
    )
    
    print("Advanced visualizations saved in outputs/")

def main():
    """Run visualization examples."""
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
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
    
    # Create output directory
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Run examples
    print("Running visualization examples...")
    
    print("\n1. Basic Structure Visualization")
    basic_structure_visualization(models[0], sequence)
    
    print("\n2. Ensemble Visualization")
    ensemble_visualization(models, sequence)
    
    print("\n3. Advanced Visualization")
    advanced_visualization(models[0], sequence)
    
    print("\nAll visualizations completed. Check the outputs/ directory.")

if __name__ == "__main__":
    main() 