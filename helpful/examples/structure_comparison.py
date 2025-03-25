"""
Example script demonstrating structure comparison capabilities of the Proteus framework.
"""

import numpy as np
from src.analysis.structure import (
    calculate_rmsd,
    calculate_tm_score,
    calculate_contact_map_overlap,
    calculate_secondary_structure_similarity,
    align_structures,
    compare_structures
)
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.visualization import (
    plot_structure_comparison,
    plot_rmsd_distribution,
    plot_contact_map,
    VisualizationConfig
)

def basic_structure_comparison(structure1, structure2):
    """Demonstrate basic structure comparison metrics."""
    # Calculate RMSD
    rmsd = calculate_rmsd(structure1, structure2)
    print(f"\nRMSD between structures: {rmsd:.3f} Å")
    
    # Calculate TM-score
    tm_score = calculate_tm_score(structure1, structure2)
    print(f"TM-score: {tm_score:.3f}")
    
    # Calculate contact map overlap
    cmo = calculate_contact_map_overlap(structure1, structure2)
    print(f"Contact map overlap: {cmo:.3f}")
    
    # Calculate secondary structure similarity
    ss_sim = calculate_secondary_structure_similarity(structure1, structure2)
    print(f"Secondary structure similarity: {ss_sim:.3f}")
    
    return {
        'rmsd': rmsd,
        'tm_score': tm_score,
        'contact_map_overlap': cmo,
        'ss_similarity': ss_sim
    }

def advanced_structure_comparison(structure1, structure2):
    """Demonstrate advanced structure comparison capabilities."""
    # Align structures
    aligned_structure1, aligned_structure2, transformation = align_structures(
        structure1,
        structure2,
        method='kabsch'  # or 'quaternion'
    )
    
    # Compare aligned structures
    comparison_results = compare_structures(
        aligned_structure1,
        aligned_structure2,
        metrics=['rmsd', 'tm_score', 'contact_map', 'secondary_structure'],
        per_residue=True
    )
    
    print("\nDetailed structure comparison:")
    print(f"Global RMSD: {comparison_results['global_rmsd']:.3f} Å")
    print(f"Per-residue RMSD range: {comparison_results['per_residue_rmsd'].min():.3f} - {comparison_results['per_residue_rmsd'].max():.3f} Å")
    print(f"TM-score: {comparison_results['tm_score']:.3f}")
    print(f"Contact map correlation: {comparison_results['contact_map_correlation']:.3f}")
    print(f"Secondary structure identity: {comparison_results['ss_identity']:.3f}")
    
    return comparison_results

def visualize_comparison(structure1, structure2, comparison_results, output_dir):
    """Demonstrate visualization of structure comparisons."""
    # Configure visualization
    config = VisualizationConfig(
        figure_size=(12, 8),
        dpi=300,
        style='seaborn',
        color_palette='Set2'
    )
    
    # Plot structure comparison
    plot_structure_comparison(
        structures={
            'structure1': structure1,
            'structure2': structure2
        },
        output_file=f"{output_dir}/structure_comparison.png",
        config=config,
        show_rmsd=True,
        show_contacts=True
    )
    
    # Plot RMSD distribution
    plot_rmsd_distribution(
        comparison_results['per_residue_rmsd'],
        output_file=f"{output_dir}/rmsd_distribution.png",
        config=config,
        title="Per-residue RMSD Distribution"
    )
    
    # Plot contact maps
    plot_contact_map(
        comparison_results['contact_map1'],
        output_file=f"{output_dir}/contact_map1.png",
        config=config,
        title="Structure 1 Contact Map"
    )
    
    plot_contact_map(
        comparison_results['contact_map2'],
        output_file=f"{output_dir}/contact_map2.png",
        config=config,
        title="Structure 2 Contact Map"
    )

def compare_model_predictions(sequence):
    """Compare predictions from different models."""
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
    
    # Get predictions
    predictions = {
        f"model_{i}": model.predict_structure(sequence)
        for i, model in enumerate(models)
    }
    
    # Compare predictions
    print("\nComparing model predictions:")
    for i, (name1, pred1) in enumerate(predictions.items()):
        for name2, pred2 in list(predictions.items())[i+1:]:
            print(f"\nComparing {name1} vs {name2}:")
            metrics = basic_structure_comparison(
                pred1['structure'],
                pred2['structure']
            )
            
            # Print additional metrics
            print(f"Confidence correlation: {np.corrcoef(pred1['confidence'], pred2['confidence'])[0,1]:.3f}")
    
    return predictions

def main():
    """Run structure comparison examples."""
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    # Create output directory
    import os
    output_dir = "outputs/structure_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running structure comparison examples...")
    
    # Compare model predictions
    predictions = compare_model_predictions(sequence)
    
    # Get structures for detailed comparison
    structure1 = predictions['model_0']['structure']
    structure2 = predictions['model_1']['structure']
    
    # Run advanced comparison
    comparison_results = advanced_structure_comparison(structure1, structure2)
    
    # Visualize results
    visualize_comparison(structure1, structure2, comparison_results, output_dir)
    
    print(f"\nAll comparisons completed. Check the {output_dir}/ directory for visualizations.")

if __name__ == "__main__":
    main() 