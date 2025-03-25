"""Test configuration and fixtures."""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import numpy as np

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_model_weights():
    """Create test model weights."""
    return torch.randn(10, 10)

@pytest.fixture
def test_sequence():
    """Return a test protein sequence."""
    return 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'

@pytest.fixture
def test_pdb_file(temp_dir):
    """Create a test PDB file."""
    pdb_path = Path(temp_dir) / 'test.pdb'
    pdb_content = """ATOM      1  N   ALA A   1      27.360  44.010  48.276  1.00 35.88           N  
ATOM      2  CA  ALA A   1      27.885  44.863  47.192  1.00 35.88           C  
ATOM      3  C   ALA A   1      29.396  44.859  47.192  1.00 35.88           C  
ATOM      4  O   ALA A   1      29.923  43.748  47.192  1.00 35.88           O  
ATOM      5  CB  ALA A   1      27.360  46.276  47.192  1.00 35.88           C  
END"""
    pdb_path.write_text(pdb_content)
    return str(pdb_path)

@pytest.fixture
def test_config():
    """Return a test configuration dictionary."""
    return {
        'model_name': 'test_model',
        'model_type': 'test_type',
        'model_path': 'test_model.pt',
        'config_path': 'test_config.yaml',
        'max_cache_size': 1024 * 1024 * 1024,  # 1GB
        'max_cache_age_days': 30
    }

@pytest.fixture
def test_metadata():
    """Return test metadata dictionary."""
    return {
        'version': '1.0.0',
        'description': 'Test model',
        'author': 'Test Author',
        'date': '2024-03-23'
    }

@pytest.fixture
def test_ensemble_data():
    """Create test ensemble data."""
    return {
        'structures': [
            torch.randn(10, 3),  # 10 residues, 3 coordinates
            torch.randn(10, 3),
            torch.randn(10, 3)
        ],
        'confidences': [
            torch.rand(10),  # 10 confidence scores
            torch.rand(10),
            torch.rand(10)
        ]
    }

@pytest.fixture
def test_model_interface(temp_dir, test_model_weights):
    """Create a test model interface."""
    weights_path = os.path.join(temp_dir, 'test_model.pt')
    torch.save(test_model_weights, weights_path)
    
    from src.models import ModelInterface
    return ModelInterface(
        model_name='test_model',
        model_type='test_type',
        model_path=weights_path
    )

@pytest.fixture
def test_cache(temp_dir):
    """Create a test model cache."""
    from src.utils import ModelCache
    return ModelCache(
        cache_dir=os.path.join(temp_dir, 'cache'),
        max_size_bytes=1024 * 1024 * 1024  # 1GB
    )

@pytest.fixture
def test_version_manager(temp_dir):
    """Create a test version manager."""
    from src.utils import ModelVersionManager
    return ModelVersionManager(temp_dir)

@pytest.fixture
def test_database(temp_dir):
    """Create a test database."""
    from src.utils import DatabaseManager
    return DatabaseManager(os.path.join(temp_dir, 'test.db'))

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
            
    return MockModel()

@pytest.fixture
def mock_models():
    """Create multiple mock models for ensemble testing."""
    return {
        "model1": mock_model(),
        "model2": mock_model()
    }

@pytest.fixture
def test_data():
    """Create test data for predictions."""
    return {
        "structure1": np.random.rand(100, 3),
        "structure2": np.random.rand(100, 3),
        "sequence": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    }

@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for test files."""
    return tmp_path_factory.mktemp("test_data")