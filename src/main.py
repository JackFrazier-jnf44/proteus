import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import logging
import os
import argparse

from src.interfaces.model_interface import ModelInterface
from src.core.pdb_encoder import encode_pdb_to_distance_matrix, process_structure_files
from src.core.plotting import generate_plots
from src.core.protein_visualization import (
    plot_contact_map,
    plot_confidence_per_residue,
    plot_rmsd_distribution,
    plot_secondary_structure,
    plot_attention_weights,
    plot_embedding_space
)
from src.config import (
    DEFAULT_OUTPUT_DIR,
    PLOT_SETTINGS,
    ANALYSIS_SETTINGS,
    SUPPORTED_FORMATS,
    ERROR_MESSAGES
)
from src.logging_config import setup_logging
from src.interfaces.base_interface import BaseModelInterface
from src.interfaces.openfold_interface import OpenFoldInterface
from src.interfaces.esm_interface import ESMInterface
from src.interfaces.alphafold_interface import AlphaFoldInterface
from src.interfaces.rosettafold_interface import RoseTTAFoldInterface
from src.interfaces.colabfold_interface import ColabFoldInterface
from src.core.batch_processor import BatchProcessor, BatchConfig
from src.core.visualization import visualize_structure
from src.core.ensemble import EnsemblePredictor
from src.config import Config

logger = logging.getLogger(__name__)

def get_model_interface(model_type: str, model_path: str, device: str = "cpu") -> BaseModelInterface:
    """Get the appropriate model interface based on model type.

    Args:
        model_type: Type of model to use
        model_path: Path to model weights
        device: Device to run model on

    Returns:
        Model interface instance
    """
    model_interfaces = {
        "openfold": OpenFoldInterface,
        "esm": ESMInterface,
        "alphafold": AlphaFoldInterface,
        "rosettafold": RoseTTAFoldInterface,
        "colabfold": ColabFoldInterface,
    }
    
    if model_type not in model_interfaces:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model_interfaces[model_type](model_path, device)

def process_single_sequence(
    sequence: str,
    model_type: str,
    model_path: str,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    templates: Optional[List[Dict]] = None,
) -> Dict:
    """Process a single protein sequence.

    Args:
        sequence: Protein sequence
        model_type: Type of model to use
        model_path: Path to model weights
        output_dir: Optional output directory
        device: Device to run model on
        templates: Optional template structures

    Returns:
        Dictionary containing prediction results
    """
    model = get_model_interface(model_type, model_path, device)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    structure, confidence, distance_matrix = model.predict(sequence, templates=templates)
    
    if output_dir:
        output_path = Path(output_dir)
        structure_path = output_path / "structure.pdb"
        confidence_path = output_path / "confidence.npy"
        distance_path = output_path / "distance.npy"
        
        structure.save(str(structure_path))
        np.save(str(confidence_path), confidence)
        np.save(str(distance_path), distance_matrix)
    
    return {
        "structure": structure,
        "confidence": confidence,
        "distance_matrix": distance_matrix,
    }

def process_batch(
    sequences: List[str],
    model_type: str,
    model_path: str,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 4,
    max_workers: int = 4,
    templates: Optional[List[Dict]] = None,
    sequence_ids: Optional[List[str]] = None,
) -> Dict:
    """Process a batch of protein sequences.

    Args:
        sequences: List of protein sequences
        model_type: Type of model to use
        model_path: Path to model weights
        output_dir: Optional output directory
        device: Device to run model on
        batch_size: Size of each batch
        max_workers: Maximum number of worker threads
        templates: Optional template structures
        sequence_ids: Optional sequence identifiers

    Returns:
        Dictionary mapping sequence IDs to prediction results
    """
    model = get_model_interface(model_type, model_path, device)
    
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        use_gpu=device != "cpu",
        output_dir=output_dir,
    )
    
    processor = BatchProcessor(model, config)
    return processor.process_batch(sequences, sequence_ids, templates)

def process_directory(
    input_dir: str,
    model_type: str,
    model_path: str,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 4,
    max_workers: int = 4,
    file_pattern: str = "*.fasta",
) -> Dict:
    """Process all sequences in a directory.

    Args:
        input_dir: Directory containing sequence files
        model_type: Type of model to use
        model_path: Path to model weights
        output_dir: Optional output directory
        device: Device to run model on
        batch_size: Size of each batch
        max_workers: Maximum number of worker threads
        file_pattern: Pattern for sequence files

    Returns:
        Dictionary mapping sequence IDs to prediction results
    """
    model = get_model_interface(model_type, model_path, device)
    
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        use_gpu=device != "cpu",
        output_dir=output_dir,
    )
    
    processor = BatchProcessor(model, config)
    return processor.process_directory(input_dir, output_dir, file_pattern)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Model Analysis Framework")
    parser.add_argument("--model", required=True, help="Type of model to use")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--sequence", help="Single protein sequence to process")
    parser.add_argument("--input-dir", help="Directory containing sequence files")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device to run model on")
    parser.add_argument("--batch-size", type=int, default=4, help="Size of each batch")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--file-pattern", default="*.fasta", help="Pattern for sequence files")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.sequence:
            # Process single sequence
            results = process_single_sequence(
                args.sequence,
                args.model,
                args.model_path,
                args.output_dir,
                args.device,
            )
            logger.info("Successfully processed single sequence")
            
        elif args.input_dir:
            # Process directory
            results = process_directory(
                args.input_dir,
                args.model,
                args.model_path,
                args.output_dir,
                args.device,
                args.batch_size,
                args.max_workers,
                args.file_pattern,
            )
            logger.info(f"Successfully processed {len(results)} sequences")
            
        else:
            parser.error("Either --sequence or --input-dir must be provided")
            
    except Exception as e:
        logger.error(f"Error processing sequences: {str(e)}")
        raise

if __name__ == "__main__":
    main()