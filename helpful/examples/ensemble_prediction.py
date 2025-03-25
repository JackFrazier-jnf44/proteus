"""
Example script demonstrating various ensemble prediction capabilities.
"""

import numpy as np
from src.core.ensemble import EnsemblePredictor, EnsembleConfig, EnsembleMethod
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.visualization import plot_ensemble_comparison
from src.core.database import DatabaseManager

def basic_ensemble_example():
    """Demonstrate basic ensemble prediction with weighted averaging."""
    # Configure models
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
    
    # Configure ensemble
    config = EnsembleConfig(
        method=EnsembleMethod.WEIGHTED_AVERAGE,
        weights={"esm": 0.6, "alphafold": 0.4}
    )
    ensemble = EnsemblePredictor(config)
    
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    # Get individual predictions
    predictions = {
        "esm": models[0].predict_structure(sequence),
        "alphafold": models[1].predict_structure(sequence)
    }
    
    # Combine predictions
    result = ensemble.combine_predictions(predictions)
    
    print("\nBasic Ensemble Results:")
    print(f"Structure file: {result['structure']}")
    print(f"Average confidence: {result['confidence'].mean():.3f}")
    
    return result, predictions

def confidence_weighted_ensemble():
    """Demonstrate confidence-weighted ensemble prediction."""
    # Configure models
    models = [
        ModelInterface(BaseModelConfig(name="esm2_t33_650M_UR50D", model_type="esm")),
        ModelInterface(BaseModelConfig(name="esm2_t36_3B_UR50D", model_type="esm")),
        ModelInterface(BaseModelConfig(name="alphafold2_ptm", model_type="alphafold"))
    ]
    
    # Configure ensemble with confidence weighting
    config = EnsembleConfig(
        method=EnsembleMethod.CONFIDENCE_WEIGHTED,
        confidence_threshold=0.7  # Only use predictions with confidence > 0.7
    )
    ensemble = EnsemblePredictor(config)
    
    # Test sequences
    sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
    ]
    
    # Process multiple sequences
    for seq in sequences:
        # Get predictions from all models
        predictions = {
            f"model_{i}": model.predict_structure(seq)
            for i, model in enumerate(models)
        }
        
        # Combine predictions
        result = ensemble.combine_predictions(predictions)
        
        print(f"\nConfidence-weighted results for sequence length {len(seq)}:")
        print(f"Structure file: {result['structure']}")
        print(f"Average confidence: {result['confidence'].mean():.3f}")
        
        # Store results in database
        db = DatabaseManager("predictions.db")
        db.store_prediction(result, metadata={
            "sequence": seq,
            "method": "confidence_weighted_ensemble",
            "models_used": len(predictions)
        })

def dynamic_ensemble():
    """Demonstrate dynamic ensemble prediction with weight adaptation."""
    # Configure models
    models = [
        ModelInterface(BaseModelConfig(name="esm2_t33_650M_UR50D", model_type="esm")),
        ModelInterface(BaseModelConfig(name="alphafold2_ptm", model_type="alphafold"))
    ]
    
    # Configure ensemble with dynamic weighting
    config = EnsembleConfig(
        method=EnsembleMethod.DYNAMIC_WEIGHTED,
        initial_weights={"esm": 0.5, "alphafold": 0.5},
        adaptation_rate=0.1
    )
    ensemble = EnsemblePredictor(config)
    
    # Test sequences with known structures for adaptation
    training_sequences = [
        ("MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", "5AWL_1"),
        ("KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE", "6A5J")
    ]
    
    # Train ensemble weights
    for seq, pdb_id in training_sequences:
        predictions = {
            "esm": models[0].predict_structure(seq),
            "alphafold": models[1].predict_structure(seq)
        }
        
        # Update weights based on prediction quality
        ensemble.update_weights(predictions, reference_pdb=pdb_id)
        print(f"\nUpdated weights after {pdb_id}:")
        print(ensemble.current_weights)
    
    return ensemble

def ensemble_visualization(result, predictions):
    """Demonstrate ensemble visualization capabilities."""
    # Plot ensemble comparison
    plot_ensemble_comparison(
        predictions=predictions,
        result=result,
        output_file="ensemble_comparison.png",
        plot_config={
            "figure_size": (15, 10),
            "dpi": 300,
            "show_confidence": True,
            "show_rmsd": True
        }
    )
    
    print("\nVisualization saved as 'ensemble_comparison.png'")

def main():
    """Run ensemble prediction examples."""
    print("Running ensemble prediction examples...")
    
    # Basic weighted ensemble
    result, predictions = basic_ensemble_example()
    
    # Confidence-weighted ensemble
    confidence_weighted_ensemble()
    
    # Dynamic ensemble
    trained_ensemble = dynamic_ensemble()
    
    # Visualize results
    ensemble_visualization(result, predictions)

if __name__ == "__main__":
    main()