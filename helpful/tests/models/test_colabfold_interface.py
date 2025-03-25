"""Tests for ColabFold model interface."""

import os
import pytest
import numpy as np
from Bio.PDB import Structure

from multi_model_analysis.models.colabfold_interface import ColabFoldInterface
from multi_model_analysis.exceptions import ModelError, ValidationError


@pytest.fixture
def colabfold_interface(tmp_path):
    """Create a ColabFold interface instance for testing."""
    model_path = os.path.join(tmp_path, "colabfold_model")
    return ColabFoldInterface(
        model_path=model_path,
        device="cpu",
        num_recycles=1,
        num_ensemble=1,
        use_templates=False,
        use_amber=False,
    )


def test_initialization(colabfold_interface):
    """Test ColabFold interface initialization."""
    assert colabfold_interface.num_recycles == 1
    assert colabfold_interface.num_ensemble == 1
    assert not colabfold_interface.use_templates
    assert not colabfold_interface.use_amber
    assert colabfold_interface.device == "cpu"


def test_model_loading(colabfold_interface):
    """Test model loading functionality."""
    with pytest.raises(ModelError):
        colabfold_interface._load_model()


def test_prediction(colabfold_interface):
    """Test structure prediction."""
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    with pytest.raises(ModelError):
        colabfold_interface.predict(sequence)


def test_input_preparation(colabfold_interface):
    """Test input preparation."""
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    with pytest.raises(ValidationError):
        colabfold_interface._prepare_input(sequence)


def test_structure_processing(colabfold_interface):
    """Test structure processing."""
    outputs = {"structure": None}
    
    with pytest.raises(ModelError):
        colabfold_interface._process_structure(outputs)


def test_confidence_processing(colabfold_interface):
    """Test confidence score processing."""
    outputs = {"confidence": np.random.rand(10, 10)}
    confidence = colabfold_interface._process_confidence(outputs)
    assert isinstance(confidence, np.ndarray)
    assert confidence.shape == (10, 10)


def test_distance_matrix_processing(colabfold_interface):
    """Test distance matrix processing."""
    outputs = {"distance_matrix": np.random.rand(10, 10)}
    distance_matrix = colabfold_interface._process_distance_matrix(outputs)
    assert isinstance(distance_matrix, np.ndarray)
    assert distance_matrix.shape == (10, 10)


def test_model_info(colabfold_interface):
    """Test model information retrieval."""
    info = colabfold_interface.get_model_info()
    assert info["name"] == "ColabFold"
    assert info["version"] == "1.5.2"
    assert info["num_recycles"] == 1
    assert info["num_ensemble"] == 1
    assert not info["use_templates"]
    assert not info["use_amber"]
    assert info["device"] == "cpu"


def test_template_handling(colabfold_interface):
    """Test template handling."""
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    templates = [{"pdb": "template.pdb", "chain": "A"}]
    
    with pytest.raises(ValidationError):
        colabfold_interface._prepare_input(sequence, templates)


def test_error_handling(colabfold_interface):
    """Test error handling."""
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    # Test invalid sequence
    with pytest.raises(ValidationError):
        colabfold_interface._prepare_input("")
    
    # Test invalid templates
    with pytest.raises(ValidationError):
        colabfold_interface._prepare_input(sequence, [{"invalid": "template"}])
    
    # Test invalid outputs
    with pytest.raises(ModelError):
        colabfold_interface._process_structure({})
    with pytest.raises(ModelError):
        colabfold_interface._process_confidence({})
    with pytest.raises(ModelError):
        colabfold_interface._process_distance_matrix({}) 