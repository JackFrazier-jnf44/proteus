"""Tests for benchmarking functionality."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
from Bio.PDB import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from tests.benchmarking.metrics import ModelMetrics, StructureMetrics, PerformanceMetrics
from tests.benchmarking.datasets import ProteinEntry, BenchmarkDataset, ProteinDataset
from tests.benchmarking.performance import BenchmarkResult, PerformanceTracker

@pytest.fixture
def sample_structure():
    """Create a sample protein structure for testing."""
    structure = Structure.Structure("test")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    
    # Create a simple alpha helix-like structure
    for i in range(10):
        res = Residue.Residue((" ", i, " "), "ALA", "")
        ca = Atom.Atom("CA", (i*3.8, 0, 0), 20.0, 1.0, " ", "CA", 1, "C")
        res.add(ca)
        chain.add(res)
    
    model.add(chain)
    structure.add(model)
    return structure

@pytest.fixture
def sample_dataset(sample_structure):
    """Create a sample dataset for testing."""
    entries = [
        ProteinEntry(
            id="test1",
            sequence="AAAAAAAAA",
            structure=sample_structure,
            metadata={"source": "test"}
        ),
        ProteinEntry(
            id="test2",
            sequence="AAAAAAAAAA",
            structure=sample_structure,
            metadata={"source": "test"}
        )
    ]
    
    return ProteinDataset(
        name="test_dataset",
        description="Test dataset",
        entries=entries,
        metadata={"type": "test"}
    )

def test_model_metrics(sample_structure):
    """Test calculation of model metrics."""
    metrics = ModelMetrics.calculate(
        predicted=sample_structure,
        reference=sample_structure,
        confidence_score=0.9,
        perf_stats={"time": 1.0, "memory": 2.0}
    )
    
    assert metrics.rmsd == pytest.approx(0.0)  # Same structure
    assert metrics.confidence == 0.9
    assert metrics.processing_time == 1.0
    assert metrics.memory_usage == 2.0

def test_structure_metrics(sample_structure):
    """Test calculation of structure quality metrics."""
    metrics = StructureMetrics.calculate(sample_structure)
    
    # Basic validation of metric existence
    assert hasattr(metrics, "rama_favored")
    assert hasattr(metrics, "clash_score")
    assert hasattr(metrics, "molprobity_score")

def test_performance_metrics():
    """Test calculation of performance metrics."""
    perf_data = {
        "inference_time": 1.0,
        "preprocessing_time": 0.5,
        "postprocessing_time": 0.3,
        "peak_memory_usage": 4.2,
        "gpu_memory_usage": 2.1,
        "cpu_utilization": 75.0,
        "gpu_utilization": 85.0,
        "throughput": 10.0
    }
    
    metrics = PerformanceMetrics.calculate(perf_data)
    
    assert metrics.inference_time == 1.0
    assert metrics.peak_memory_usage == 4.2
    assert metrics.throughput == 10.0

def test_dataset_loading(tmp_path):
    """Test dataset loading functionality."""
    # Create test FASTA file
    fasta_path = tmp_path / "test.fasta"
    with open(fasta_path, "w") as f:
        f.write(">test1\nAAAAA\n>test2\nCCCCC\n")
    
    # Create test metadata
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        f.write('{"test1": {"source": "test"}, "test2": {"source": "test"}}')
    
    dataset = BenchmarkDataset.from_fasta(fasta_path, metadata_path)
    
    assert len(dataset.entries) == 2
    assert dataset.entries[0].sequence == "AAAAA"
    assert dataset.entries[1].sequence == "CCCCC"

def test_protein_dataset_operations(sample_dataset):
    """Test protein dataset operations."""
    # Test length distribution
    dist = sample_dataset.get_length_distribution()
    assert dist["min"] == 9.0
    assert dist["max"] == 10.0
    
    # Test filtering
    filtered = sample_dataset.filter_by_length(min_length=10)
    assert len(filtered.entries) == 1
    assert filtered.entries[0].id == "test2"
    
    # Test splitting
    splits = sample_dataset.split(train_ratio=0.5, val_ratio=0.25, random_seed=42)
    assert len(splits) == 3
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

def test_performance_tracking():
    """Test performance tracking functionality."""
    tracker = PerformanceTracker()
    
    # Test tracking start
    tracker.start_tracking()
    assert tracker.start_time is not None
    
    # Test memory stats update
    tracker.update_memory_stats()
    assert tracker.current_memory > 0
    
    # Test performance data collection
    data = tracker.get_performance_data()
    assert "inference_time" in data
    assert "peak_memory_usage" in data

def test_benchmark_results_saving(tmp_path):
    """Test saving and loading of benchmark results."""
    tracker = PerformanceTracker()
    
    # Create sample result
    result = BenchmarkResult(
        model_name="test_model",
        dataset_name="test_dataset",
        timestamp=123456789.0,
        model_metrics=ModelMetrics(
            rmsd=1.0, tm_score=0.8, gdt_ts=0.7, gdt_ha=0.6,
            lddt=0.9, confidence=0.85, processing_time=1.0, memory_usage=2.0
        ),
        structure_metrics=StructureMetrics(
            rama_favored=98.0, rama_allowed=2.0, rama_outliers=0.0,
            rotamer_outliers=1.0, clash_score=0.5, molprobity_score=1.2
        ),
        performance_metrics=PerformanceMetrics(
            inference_time=1.0, preprocessing_time=0.5, postprocessing_time=0.3,
            peak_memory_usage=4.2, gpu_memory_usage=2.1, cpu_utilization=75.0,
            gpu_utilization=85.0, throughput=10.0
        )
    )
    
    tracker.add_result(result)
    
    # Save results
    save_path = tmp_path / "results.json"
    tracker.save_results(save_path)
    
    # Load results
    loaded_tracker = PerformanceTracker.load_results(save_path)
    
    assert len(loaded_tracker.results) == 1
    loaded_result = loaded_tracker.results[0]
    assert loaded_result.model_name == "test_model"
    assert loaded_result.model_metrics.rmsd == 1.0
    assert loaded_result.performance_metrics.throughput == 10.0 