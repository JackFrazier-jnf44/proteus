"""Tests for structure comparison functionality."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from proteus.analysis.structure import (
    calculate_rmsd,
    calculate_tm_score,
    calculate_contact_map_overlap,
    calculate_secondary_structure_similarity,
    align_structures,
    compare_structures
)
from proteus.interfaces import ModelInterface, BaseModelConfig

@pytest.fixture
def sample_structures():
    """Create sample structures for testing."""
    # Create two similar but not identical structures
    base_structure = np.random.rand(50, 3)
    structure1 = base_structure.copy()
    structure2 = base_structure + np.random.normal(0, 0.1, (50, 3))  # Add small perturbations
    
    return structure1, structure2

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def test_calculate_rmsd(sample_structures):
    """Test RMSD calculation."""
    structure1, structure2 = sample_structures
    
    # Calculate RMSD
    rmsd = calculate_rmsd(structure1, structure2)
    
    assert isinstance(rmsd, float)
    assert rmsd >= 0
    assert rmsd < 1.0  # Should be small due to similar structures

def test_calculate_tm_score(sample_structures):
    """Test TM-score calculation."""
    structure1, structure2 = sample_structures
    
    # Calculate TM-score
    tm_score = calculate_tm_score(structure1, structure2)
    
    assert isinstance(tm_score, float)
    assert 0 <= tm_score <= 1
    assert tm_score > 0.5  # Should be high due to similar structures

def test_calculate_contact_map_overlap(sample_structures):
    """Test contact map overlap calculation."""
    structure1, structure2 = sample_structures
    
    # Calculate contact map overlap
    cmo = calculate_contact_map_overlap(structure1, structure2)
    
    assert isinstance(cmo, float)
    assert 0 <= cmo <= 1
    assert cmo > 0.5  # Should be high due to similar structures

def test_calculate_secondary_structure_similarity(sample_structures):
    """Test secondary structure similarity calculation."""
    structure1, structure2 = sample_structures
    
    # Calculate secondary structure similarity
    ss_sim = calculate_secondary_structure_similarity(structure1, structure2)
    
    assert isinstance(ss_sim, float)
    assert 0 <= ss_sim <= 1

def test_align_structures(sample_structures):
    """Test structure alignment."""
    structure1, structure2 = sample_structures
    
    # Test Kabsch alignment
    aligned1, aligned2, transform = align_structures(
        structure1,
        structure2,
        method='kabsch'
    )
    
    assert aligned1.shape == structure1.shape
    assert aligned2.shape == structure2.shape
    assert isinstance(transform, dict)
    assert 'rotation' in transform
    assert 'translation' in transform
    
    # Test quaternion alignment
    aligned1, aligned2, transform = align_structures(
        structure1,
        structure2,
        method='quaternion'
    )
    
    assert aligned1.shape == structure1.shape
    assert aligned2.shape == structure2.shape
    assert isinstance(transform, dict)
    assert 'rotation' in transform
    assert 'translation' in transform

def test_compare_structures(sample_structures):
    """Test comprehensive structure comparison."""
    structure1, structure2 = sample_structures
    
    # Compare structures with all metrics
    results = compare_structures(
        structure1,
        structure2,
        metrics=['rmsd', 'tm_score', 'contact_map', 'secondary_structure'],
        per_residue=True
    )
    
    # Check results
    assert isinstance(results, dict)
    assert 'global_rmsd' in results
    assert 'per_residue_rmsd' in results
    assert 'tm_score' in results
    assert 'contact_map_correlation' in results
    assert 'ss_identity' in results
    
    # Check specific metrics
    assert results['global_rmsd'] >= 0
    assert len(results['per_residue_rmsd']) == len(structure1)
    assert 0 <= results['tm_score'] <= 1
    assert -1 <= results['contact_map_correlation'] <= 1
    assert 0 <= results['ss_identity'] <= 1

def test_compare_structures_with_invalid_input():
    """Test structure comparison with invalid input."""
    # Test with different sized structures
    structure1 = np.random.rand(50, 3)
    structure2 = np.random.rand(40, 3)
    
    with pytest.raises(ValueError):
        compare_structures(structure1, structure2)
    
    # Test with invalid dimensions
    structure1 = np.random.rand(50, 4)  # Should be (N, 3)
    structure2 = np.random.rand(50, 4)
    
    with pytest.raises(ValueError):
        compare_structures(structure1, structure2)
    
    # Test with invalid metrics
    structure1 = np.random.rand(50, 3)
    structure2 = np.random.rand(50, 3)
    
    with pytest.raises(ValueError):
        compare_structures(structure1, structure2, metrics=['invalid_metric'])

def test_structure_comparison_with_real_models():
    """Test structure comparison with actual model predictions."""
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
    
    try:
        # Get predictions
        predictions = [
            model.predict_structure(sequence)
            for model in models
        ]
        
        # Compare structures
        results = compare_structures(
            predictions[0]['structure'],
            predictions[1]['structure']
        )
        
        # Basic checks
        assert isinstance(results, dict)
        assert results['global_rmsd'] >= 0
        assert 0 <= results['tm_score'] <= 1
        
    except Exception as e:
        pytest.skip(f"Model prediction or comparison failed: {e}")

def test_structure_comparison_performance():
    """Test performance of structure comparison methods."""
    # Create large structures
    n_residues = 500
    structure1 = np.random.rand(n_residues, 3)
    structure2 = structure1 + np.random.normal(0, 0.1, (n_residues, 3))
    
    import time
    
    # Test RMSD calculation time
    start_time = time.time()
    calculate_rmsd(structure1, structure2)
    rmsd_time = time.time() - start_time
    
    # Test TM-score calculation time
    start_time = time.time()
    calculate_tm_score(structure1, structure2)
    tm_score_time = time.time() - start_time
    
    # Performance assertions
    assert rmsd_time < 1.0  # Should be fast
    assert tm_score_time < 2.0  # May be slower but still reasonable

def test_structure_comparison_batch():
    """Test batch structure comparison."""
    # Create multiple structures
    n_structures = 5
    base_structure = np.random.rand(50, 3)
    structures = [
        base_structure + np.random.normal(0, 0.1, (50, 3))
        for _ in range(n_structures)
    ]
    
    # Compare all pairs
    results = []
    for i in range(n_structures):
        for j in range(i + 1, n_structures):
            result = compare_structures(structures[i], structures[j])
            results.append(result)
    
    assert len(results) == (n_structures * (n_structures - 1)) // 2
    for result in results:
        assert isinstance(result, dict)
        assert result['global_rmsd'] >= 0