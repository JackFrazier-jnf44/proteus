"""Batch processing utilities for multiple protein structures."""

import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

import numpy as np
from Bio.PDB import Structure

from src.interfaces.base_interface import BaseModelInterface
from src.exceptions import ModelError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 4
    max_workers: int = 4
    use_gpu: bool = True
    output_dir: Optional[str] = None
    save_intermediate: bool = True
    template_dir: Optional[str] = None
    error_handling: str = "skip"  # Options: "skip", "raise", "log"

class BatchProcessor:
    """Handles batch processing of multiple protein structures."""

    def __init__(
        self,
        model_interface: BaseModelInterface,
        config: Optional[BatchConfig] = None,
    ):
        """Initialize batch processor.

        Args:
            model_interface: Model interface instance
            config: Optional batch processing configuration
        """
        self.model = model_interface
        self.config = config or BatchConfig()
        self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """Setup output directory for batch processing results."""
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)

    def process_batch(
        self,
        sequences: List[str],
        sequence_ids: Optional[List[str]] = None,
        templates: Optional[List[Dict]] = None,
    ) -> Dict[str, Tuple[Structure, np.ndarray, np.ndarray]]:
        """Process a batch of protein sequences.

        Args:
            sequences: List of protein sequences
            sequence_ids: Optional list of sequence identifiers
            templates: Optional list of template structures

        Returns:
            Dictionary mapping sequence IDs to prediction results
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        results = {}
        failed_sequences = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_seq = {
                executor.submit(
                    self._process_single,
                    seq,
                    seq_id,
                    templates[i] if templates and i < len(templates) else None,
                ): seq_id
                for i, (seq, seq_id) in enumerate(zip(sequences, sequence_ids))
            }

            for future in as_completed(future_to_seq):
                seq_id = future_to_seq[future]
                try:
                    result = future.result()
                    results[seq_id] = result
                except Exception as e:
                    failed_sequences.append((seq_id, str(e)))
                    if self.config.error_handling == "raise":
                        raise
                    elif self.config.error_handling == "log":
                        logger.error(f"Failed to process sequence {seq_id}: {str(e)}")

        if failed_sequences:
            logger.warning(f"Failed to process {len(failed_sequences)} sequences")
            for seq_id, error in failed_sequences:
                logger.warning(f"Sequence {seq_id}: {error}")

        return results

    def _process_single(
        self,
        sequence: str,
        sequence_id: str,
        template: Optional[Dict] = None,
    ) -> Tuple[Structure, np.ndarray, np.ndarray]:
        """Process a single protein sequence.

        Args:
            sequence: Protein sequence
            sequence_id: Sequence identifier
            template: Optional template structure

        Returns:
            Tuple containing predicted structure, confidence scores, and distance matrix
        """
        try:
            # Run prediction
            structure, confidence, distance_matrix = self.model.predict(
                sequence,
                templates=[template] if template else None,
            )

            # Save intermediate results if configured
            if self.config.save_intermediate and self.config.output_dir:
                self._save_results(
                    sequence_id,
                    structure,
                    confidence,
                    distance_matrix,
                )

            return structure, confidence, distance_matrix

        except Exception as e:
            if self.config.error_handling == "raise":
                raise
            elif self.config.error_handling == "log":
                logger.error(f"Error processing sequence {sequence_id}: {str(e)}")
            return None

    def _save_results(
        self,
        sequence_id: str,
        structure: Structure,
        confidence: np.ndarray,
        distance_matrix: np.ndarray,
    ) -> None:
        """Save prediction results to files.

        Args:
            sequence_id: Sequence identifier
            structure: Predicted structure
            confidence: Confidence scores
            distance_matrix: Distance matrix
        """
        output_dir = Path(self.config.output_dir)
        
        # Save structure
        structure_path = output_dir / f"{sequence_id}_structure.pdb"
        with open(structure_path, "w") as f:
            structure.save(f)

        # Save confidence scores
        confidence_path = output_dir / f"{sequence_id}_confidence.npy"
        np.save(confidence_path, confidence)

        # Save distance matrix
        distance_path = output_dir / f"{sequence_id}_distance.npy"
        np.save(distance_path, distance_matrix)

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        file_pattern: str = "*.fasta",
    ) -> Dict[str, Tuple[Structure, np.ndarray, np.ndarray]]:
        """Process all sequences in a directory.

        Args:
            input_dir: Directory containing sequence files
            output_dir: Optional output directory
            file_pattern: Pattern for sequence files

        Returns:
            Dictionary mapping sequence IDs to prediction results
        """
        if output_dir:
            self.config.output_dir = output_dir
            self._setup_output_dir()

        input_path = Path(input_dir)
        sequence_files = list(input_path.glob(file_pattern))
        
        sequences = []
        sequence_ids = []
        
        for file_path in sequence_files:
            with open(file_path, "r") as f:
                # Simple FASTA parsing - can be enhanced based on needs
                for line in f:
                    if line.startswith(">"):
                        sequence_ids.append(line[1:].strip())
                    else:
                        sequences.append(line.strip())

        return self.process_batch(sequences, sequence_ids) 