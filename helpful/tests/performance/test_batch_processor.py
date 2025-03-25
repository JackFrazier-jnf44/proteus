"""Tests for batch processing utilities."""

import os
import pytest
import numpy as np
from pathlib import Path
from Bio.PDB import Structure

from multi_model_analysis.utils.batch_processor import BatchProcessor, BatchConfig
from multi_model_analysis.models.base_interface import BaseModelInterface
from multi_model_analysis.exceptions import ModelError, ValidationError


class MockModelInterface(BaseModelInterface):
    """Mock model interface for testing."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__(model_path, device)
        self.predictions = {}

    def predict(
        self,
        sequence: str,
        templates: list = None,
        **kwargs,
    ) -> tuple[Structure, np.ndarray, np.ndarray]:
        """Mock prediction method."""
        if sequence in self.predictions:
            return self.predictions[sequence]
        
        # Create mock outputs
        structure = Structure("mock")
        confidence = np.random.rand(10, 10)
        distance_matrix = np.random.rand(10, 10)
        
        result = (structure, confidence, distance_matrix)
        self.predictions[sequence] = result
        return result


@pytest.fixture
def mock_model(tmp_path):
    """Create a mock model interface."""
    return MockModelInterface(str(tmp_path))


@pytest.fixture
def batch_config(tmp_path):
    """Create a batch configuration."""
    return BatchConfig(
        batch_size=2,
        max_workers=2,
        use_gpu=False,
        output_dir=str(tmp_path),
        save_intermediate=True,
    )


@pytest.fixture
def batch_processor(mock_model, batch_config):
    """Create a batch processor instance."""
    return BatchProcessor(mock_model, batch_config)


def test_initialization(batch_processor, mock_model, batch_config):
    """Test batch processor initialization."""
    assert batch_processor.model == mock_model
    assert batch_processor.config == batch_config


def test_process_batch(batch_processor):
    """Test processing a batch of sequences."""
    sequences = ["MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"]
    sequence_ids = ["seq1", "seq2"]
    
    results = batch_processor.process_batch(sequences, sequence_ids)
    
    assert len(results) == 2
    assert "seq1" in results
    assert "seq2" in results
    assert isinstance(results["seq1"], tuple)
    assert len(results["seq1"]) == 3


def test_process_batch_with_templates(batch_processor):
    """Test processing a batch with templates."""
    sequences = ["MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"]
    sequence_ids = ["seq1"]
    templates = [{"pdb": "template.pdb", "chain": "A"}]
    
    results = batch_processor.process_batch(sequences, sequence_ids, templates)
    
    assert len(results) == 1
    assert "seq1" in results


def test_process_batch_error_handling(batch_processor):
    """Test error handling in batch processing."""
    # Modify mock model to raise an error for a specific sequence
    batch_processor.model.predictions["error_seq"] = None
    
    sequences = ["MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", "error_seq"]
    sequence_ids = ["seq1", "error_seq"]
    
    results = batch_processor.process_batch(sequences, sequence_ids)
    
    assert len(results) == 1
    assert "seq1" in results
    assert "error_seq" not in results


def test_save_results(batch_processor, tmp_path):
    """Test saving prediction results."""
    sequence_id = "test_seq"
    structure = Structure("test")
    confidence = np.random.rand(10, 10)
    distance_matrix = np.random.rand(10, 10)
    
    batch_processor._save_results(sequence_id, structure, confidence, distance_matrix)
    
    assert (tmp_path / f"{sequence_id}_structure.pdb").exists()
    assert (tmp_path / f"{sequence_id}_confidence.npy").exists()
    assert (tmp_path / f"{sequence_id}_distance.npy").exists()


def test_process_directory(batch_processor, tmp_path):
    """Test processing sequences from a directory."""
    # Create test FASTA files
    fasta_dir = tmp_path / "fasta_files"
    fasta_dir.mkdir()
    
    with open(fasta_dir / "seq1.fasta", "w") as f:
        f.write(">seq1\nMLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n")
    
    with open(fasta_dir / "seq2.fasta", "w") as f:
        f.write(">seq2\nMLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n")
    
    results = batch_processor.process_directory(str(fasta_dir))
    
    assert len(results) == 2
    assert "seq1" in results
    assert "seq2" in results


def test_process_directory_with_output_dir(batch_processor, tmp_path):
    """Test processing directory with custom output directory."""
    # Create test FASTA file
    fasta_dir = tmp_path / "fasta_files"
    fasta_dir.mkdir()
    
    with open(fasta_dir / "seq1.fasta", "w") as f:
        f.write(">seq1\nMLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF\n")
    
    output_dir = tmp_path / "output"
    results = batch_processor.process_directory(
        str(fasta_dir),
        output_dir=str(output_dir),
    )
    
    assert len(results) == 1
    assert "seq1" in results
    assert output_dir.exists() 