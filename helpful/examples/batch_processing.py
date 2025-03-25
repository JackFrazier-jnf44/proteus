"""
Example script demonstrating batch processing capabilities of the Proteus framework.
"""

import os
from pathlib import Path
import logging
from src.core.batch import (
    BatchProcessor,
    BatchConfig,
    DistributedBatchProcessor
)
from src.interfaces import ModelInterface, BaseModelConfig
from src.core.logging import setup_logging
from src.main import process_batch, process_directory

def basic_batch_processing():
    """Demonstrate basic batch processing functionality."""
    # Initialize processor
    processor = BatchProcessor(
        max_batch_size=10,
        num_workers=4
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
    )
    
    # Sample sequences
    sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
    ]
    
    # Process sequences
    results = processor.process_batch(
        model=model,
        sequences=sequences,
        prediction_type="structure"
    )
    
    # Print results
    print("\nBasic batch processing results:")
    for i, (sequence, result) in enumerate(zip(sequences, results)):
        print(f"\nSequence {i+1} ({len(sequence)} residues):")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing time: {result['processing_time']:.2f}s")

def batch_processing_with_templates():
    """Example of processing a batch with templates."""
    # Example sequences
    sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    ]
    
    # Optional sequence IDs
    sequence_ids = ["seq1", "seq2", "seq3"]
    
    # Optional templates
    templates = [
        {"pdb": "template1.pdb", "chain": "A"},
        {"pdb": "template2.pdb", "chain": "A"},
        None,
    ]
    
    # Process batch
    results = process_batch(
        sequences=sequences,
        model_type="openfold",  # or "esm", "alphafold", "rosettafold", "colabfold"
        model_path="path/to/model/weights",
        output_dir="output/batch_results",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        batch_size=2,
        max_workers=2,
        templates=templates,
        sequence_ids=sequence_ids,
    )
    
    print("\nBatch processing with templates results:")
    for seq_id, result in results.items():
        print(f"Results for {seq_id}:")
        print(f"  Structure: {result['structure']}")
        print(f"  Confidence shape: {result['confidence'].shape}")
        print(f"  Distance matrix shape: {result['distance_matrix'].shape}")

def directory_batch_processing():
    """Example of processing sequences from a directory."""
    # Create example FASTA files
    input_dir = Path("example_sequences")
    input_dir.mkdir(exist_ok=True)
    
    # Create some example FASTA files
    fasta_files = [
        (input_dir / "seq1.fasta", ">seq1\nMLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n"),
        (input_dir / "seq2.fasta", ">seq2\nMLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n"),
        (input_dir / "seq3.fasta", ">seq3\nMLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n"),
    ]
    
    for file_path, content in fasta_files:
        with open(file_path, "w") as f:
            f.write(content)
    
    # Process directory
    results = process_directory(
        input_dir=str(input_dir),
        model_type="openfold",  # or "esm", "alphafold", "rosettafold", "colabfold"
        model_path="path/to/model/weights",
        output_dir="output/directory_results",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        batch_size=2,
        max_workers=2,
        file_pattern="*.fasta",
    )
    
    print("\nDirectory batch processing results:")
    for seq_id, result in results.items():
        print(f"Results for {seq_id}:")
        print(f"  Structure: {result['structure']}")
        print(f"  Confidence shape: {result['confidence'].shape}")
        print(f"  Distance matrix shape: {result['distance_matrix'].shape}")

def advanced_batch_configuration():
    """Demonstrate advanced batch processing configuration."""
    # Configure batch processing
    config = BatchConfig(
        max_batch_size=10,
        num_workers=4,
        gpu_ids=[0],  # Use first GPU
        memory_limit_gb=32,
        timeout_per_sequence=300,
        retry_attempts=3,
        prefetch_batches=2,
        cleanup_interval=60,
        save_intermediate=True,
        template_dir="templates",
        error_handling="log"
    )
    
    # Initialize processor with config
    processor = BatchProcessor(config)
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="alphafold2_ptm",
            model_type="alphafold"
        )
    )
    
    # Generate test sequences
    sequences = [
        "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(length)])
        for length in [50, 100, 150, 200]
    ]
    
    # Define callback for monitoring
    def batch_callback(batch_id, status):
        print(f"Batch {batch_id}: {status}")
    
    # Process sequences with monitoring
    results = processor.process_batch(
        model=model,
        sequences=sequences,
        callbacks=[batch_callback],
        prediction_type="structure"
    )
    
    # Print detailed results
    print("\nAdvanced batch processing results:")
    for i, (sequence, result) in enumerate(zip(sequences, results)):
        print(f"\nSequence {i+1}:")
        print(f"Length: {len(sequence)}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Memory usage: {result['memory_usage_gb']:.2f} GB")
        if 'gpu_memory_usage_gb' in result:
            print(f"GPU memory usage: {result['gpu_memory_usage_gb']:.2f} GB")

def batch_processing_with_recovery():
    """Demonstrate batch processing with recovery capabilities."""
    # Configure processor with recovery options
    processor = BatchProcessor(
        recovery_strategy="continue",
        checkpoint_interval=2
    )
    
    # Create model
    model = ModelInterface(
        BaseModelConfig(
            name="rosettafold",
            model_type="rosetta"
        )
    )
    
    # Create sequences
    sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
        "MDSKGSSQKGSRLLLLLVVSNLLLCQGVVSTPVCPNGPGNCQVSLRDLFDRAVMVSHYIHDLSSEMFNEFDKRYAQGKGFITMALNSCHTSSLPTPEDKEQAQQTHHEVLMSLILGLLRSWNDPLYHLVTEVRGMKGAPDAILSRAIEIEEENKRLLEGMEMIFGQVIPGAKETEPYPVWSGLPSLQTKDEDARYSAFYNLLHCLRRDSSKIDTYLKLLNCRIIYNNNC"
    ]
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "batch_checkpoint.pkl"
    
    try:
        # Process with recovery
        results = processor.process_batch_with_recovery(
            model=model,
            sequences=sequences,
            checkpoint_file=checkpoint_file
        )
        
        print("\nBatch processing with recovery completed successfully")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("Recovering from checkpoint...")
        
        # Recover from checkpoint
        results = processor.recover_from_checkpoint(checkpoint_file)
    
    finally:
        # Cleanup checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()

def distributed_batch_processing():
    """Demonstrate distributed batch processing."""
    try:
        # Initialize distributed processor
        dist_processor = DistributedBatchProcessor(
            scheduler="dask",
            num_workers=4,
            worker_resources={
                "memory": "8GB",
                "gpu": 1
            }
        )
        
        # Create model
        model = ModelInterface(
            BaseModelConfig(
                name="esm2_t33_650M_UR50D",
                model_type="esm"
            )
        )
        
        # Generate sequences
        sequences = [
            "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(length)])
            for length in range(50, 251, 50)
        ]
        
        # Process sequences
        results = dist_processor.process_batch(
            model=model,
            sequences=sequences,
            prediction_type="structure"
        )
        
        print("\nDistributed batch processing results:")
        for i, (sequence, result) in enumerate(zip(sequences, results)):
            print(f"\nSequence {i+1}:")
            print(f"Length: {len(sequence)}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Worker: {result['worker_id']}")
        
    except ImportError:
        print("\nSkipped distributed processing example (dask not available)")
    except Exception as e:
        print(f"\nError during distributed processing: {e}")

def main():
    """Run batch processing examples."""
    # Setup logging
    setup_logging(level="INFO")
    
    print("Running batch processing examples...")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    print("\n1. Basic Batch Processing")
    basic_batch_processing()
    
    print("\n2. Batch Processing with Templates")
    batch_processing_with_templates()
    
    print("\n3. Directory Batch Processing")
    directory_batch_processing()
    
    print("\n4. Advanced Batch Configuration")
    advanced_batch_configuration()
    
    print("\n5. Batch Processing with Recovery")
    batch_processing_with_recovery()
    
    print("\n6. Distributed Batch Processing")
    distributed_batch_processing()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 