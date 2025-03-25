# Testing and Examples Documentation

This document provides detailed information about the test suite and example code for the Multi-Model Protein Structure Analysis Framework.

## Test Suite Overview

The test suite is located in the `tests/` directory and covers all major components of the framework. Tests are written using pytest and include unit tests, integration tests, and example usage tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_model_interface.py

# Run tests with verbose output
python -m pytest -v tests/
```

## Test Files

### test_model_interface.py

Tests for the model interface implementation.

```python
from src.core import *  # Replace with specific imports
from src.config import *  # Replace with specific imports

def test_model_interface_initialization():
    """Test model interface initialization with different configurations."""
    configs = [
        ModelConfig(
            name="esm",
            model_type="esm",
            output_format="pdb"
        ),
        ModelConfig(
            name="openfold",
            model_type="openfold",
            output_format="pdb"
        )
    ]
    interface = ModelInterface(configs)
    assert len(interface.models) == 2
    assert "esm" in interface.models
    assert "openfold" in interface.models

def test_model_invocation():
    """Test model invocation with a sample sequence."""
    config = ModelConfig(
        name="esm",
        model_type="esm",
        output_format="pdb"
    )
    interface = ModelInterface([config])
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    outputs = interface.invoke_models(sequence, "test_output")
    assert outputs["esm"]["structure"] is not None
    assert outputs["esm"]["embeddings"] is not None
```

### test_ensemble_and_versioning.py

Tests for ensemble methods and model versioning.

```python
from src.core import *  # Replace with specific imports
from src.config import *  # Replace with specific imports

def test_ensemble_methods():
    """Test different ensemble methods."""
    config = EnsembleConfig(
        method=EnsembleMethod.WEIGHTED_AVERAGE,
        weights={"model1": 0.6, "model2": 0.4}
    )
    ensemble = Ensemble(config)
    predictions = {
        "model1": np.random.rand(100, 3),
        "model2": np.random.rand(100, 3)
    }
    result = ensemble.combine_predictions(predictions)
    assert result.shape == (100, 3)

def test_model_versioning():
    """Test model version management."""
    version_manager = ModelVersionManager("test_versions")
    version_manager.add_version("model1", "v1.0", "path/to/v1.0")
    version_manager.add_version("model1", "v2.0", "path/to/v2.0")
    versions = version_manager.get_versions("model1")
    assert len(versions) == 2
    assert "v1.0" in versions
    assert "v2.0" in versions
```

### test_database_manager.py

Tests for database operations.

```python
from src.core import *  # Replace with specific imports
from src.config import *  # Replace with specific imports

def test_database_operations():
    """Test database operations for storing and retrieving results."""
    db = DatabaseManager("test.db")
    result = {
        "sequence": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "model": "esm",
        "structure": "path/to/structure.pdb"
    }
    db.store_result(result)
    retrieved = db.get_result(result["sequence"], result["model"])
    assert retrieved["structure"] == result["structure"]
```

### test_plotting.py

Tests for visualization functions.

```python
from src.core import *  # Replace with specific imports
from src.config import *  # Replace with specific imports

def test_plot_generation():
    """Test generation of various plots."""
    # Test contact map plotting
    distance_matrix = np.random.rand(100, 100)
    plot_contact_map(distance_matrix, "test_contact_map.png")
    assert os.path.exists("test_contact_map.png")

    # Test confidence plot
    confidence_scores = np.random.rand(100)
    plot_confidence_per_residue(confidence_scores, "test_confidence.png")
    assert os.path.exists("test_confidence.png")
```

## Example Usage

### Basic Model Usage

```python
from src.models.model_interface import ModelInterface, ModelConfig
from src.utils.ensemble import EnsembleConfig, EnsembleMethod

# Configure models
configs = [
    ModelConfig(
        name="esm",
        model_type="esm",
        output_format="pdb",
        model_name="esm2_t36_3B_UR50D"
    ),
    ModelConfig(
        name="openfold",
        model_type="openfold",
        output_format="pdb"
    )
]

# Initialize interface
interface = ModelInterface(configs)

# Analyze sequence
sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
outputs = interface.invoke_models(sequence, "output_dir")

# Process results
for model_name, result in outputs.items():
    print(f"Model: {model_name}")
    print(f"Structure file: {result['structure']}")
    print(f"Embedding shape: {result['embeddings'].shape}")
```

### Ensemble Analysis

```python
from src.utils.ensemble import Ensemble, EnsembleConfig, EnsembleMethod

# Configure ensemble
config = EnsembleConfig(
    method=EnsembleMethod.WEIGHTED_AVERAGE,
    weights={"esm": 0.6, "openfold": 0.4}
)
ensemble = Ensemble(config)

# Combine predictions
combined_structure = ensemble.combine_predictions(outputs)

# Save combined structure
save_structure(combined_structure, "ensemble_output.pdb")
```

### Visualization

```python
from src.utils.plotting import (
    plot_contact_map,
    plot_confidence_per_residue,
    plot_rmsd_distribution
)

# Generate contact map
plot_contact_map(
    distance_matrix=outputs["esm"]["distance_matrix"],
    sequence=sequence,
    output_file="contact_map.png"
)

# Plot confidence scores
plot_confidence_per_residue(
    confidence_scores=outputs["esm"]["confidence_scores"],
    sequence=sequence,
    output_file="confidence.png"
)

# Plot RMSD distribution
plot_rmsd_distribution(
    structures=[outputs["esm"]["structure"], outputs["openfold"]["structure"]],
    output_file="rmsd_distribution.png"
)
```

## Test Data

The test suite uses a set of sample protein sequences and structures located in `tests/data/`:

- `sample_sequences.fasta`: Contains test protein sequences
- `sample_structures/`: Directory containing sample PDB files
- `test_embeddings.npy`: Sample embedding data for testing

## Writing New Tests

When adding new tests:

1. Create a new test file in the `tests/` directory
2. Use descriptive test names and docstrings
3. Include both positive and negative test cases
4. Mock external dependencies when appropriate
5. Add test data to `tests/data/` if needed

Example:

```python
def test_new_feature():
    """Test description of the new feature."""
    # Setup
    test_input = "test_data"
    
    # Test positive case
    result = new_feature(test_input)
    assert result is not None
    assert result.shape == expected_shape
    
    # Test negative case
    with pytest.raises(ValueError):
        new_feature(None)
```

## Continuous Integration

The test suite is automatically run on:

- Pull requests
- Merges to main branch
- Nightly builds

Test results and coverage reports are available in the CI/CD dashboard.
