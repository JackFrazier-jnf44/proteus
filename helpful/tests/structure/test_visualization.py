import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from src.core.visualization import (
    plot_rmsd_distribution,
    plot_tm_score_distribution,
    plot_contact_map,
    plot_secondary_structure,
    plot_dihedral_angles,
    plot_structure_alignment,
    plot_ensemble_diversity
)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'rmsd_values': np.random.normal(2.0, 0.5, 100),
        'tm_scores': np.random.normal(0.8, 0.1, 100),
        'contact_map': np.random.rand(50, 50),
        'secondary_structure': np.random.choice(['H', 'E', 'C'], 50),
        'dihedral_angles': np.random.rand(50, 2) * 360,
        'structure_coords': np.random.rand(50, 3),
        'ensemble_coords': np.random.rand(5, 50, 3)
    }

def test_plot_rmsd_distribution(temp_dir, sample_data):
    """Test RMSD distribution plotting."""
    output_file = temp_dir / 'rmsd_dist.png'
    plot_rmsd_distribution(sample_data['rmsd_values'], output_file)
    assert output_file.exists()

def test_plot_tm_score_distribution(temp_dir, sample_data):
    """Test TM-score distribution plotting."""
    output_file = temp_dir / 'tm_score_dist.png'
    plot_tm_score_distribution(sample_data['tm_scores'], output_file)
    assert output_file.exists()

def test_plot_contact_map(temp_dir, sample_data):
    """Test contact map plotting."""
    output_file = temp_dir / 'contact_map.png'
    plot_contact_map(sample_data['contact_map'], output_file)
    assert output_file.exists()

def test_plot_secondary_structure(temp_dir, sample_data):
    """Test secondary structure plotting."""
    output_file = temp_dir / 'secondary_structure.png'
    plot_secondary_structure(sample_data['secondary_structure'], output_file)
    assert output_file.exists()

def test_plot_dihedral_angles(temp_dir, sample_data):
    """Test dihedral angles plotting."""
    output_file = temp_dir / 'dihedral_angles.png'
    plot_dihedral_angles(sample_data['dihedral_angles'], output_file)
    assert output_file.exists()

def test_plot_structure_alignment(temp_dir, sample_data):
    """Test structure alignment plotting."""
    output_file = temp_dir / 'structure_alignment.png'
    plot_structure_alignment(sample_data['structure_coords'], output_file)
    assert output_file.exists()

def test_plot_ensemble_diversity(temp_dir, sample_data):
    """Test ensemble diversity plotting."""
    output_file = temp_dir / 'ensemble_diversity.png'
    plot_ensemble_diversity(sample_data['ensemble_coords'], output_file)
    assert output_file.exists()

def test_plot_with_invalid_data(temp_dir):
    """Test plotting with invalid data."""
    # Test with empty arrays
    with pytest.raises(ValueError):
        plot_rmsd_distribution(np.array([]), temp_dir / 'empty.png')
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        plot_contact_map(np.random.rand(10, 20), temp_dir / 'invalid.png')
    
    # Test with invalid secondary structure labels
    with pytest.raises(ValueError):
        plot_secondary_structure(['X'], temp_dir / 'invalid.png')
    
    # Test with invalid dihedral angles
    with pytest.raises(ValueError):
        plot_dihedral_angles(np.random.rand(10, 3), temp_dir / 'invalid.png')

def test_plot_with_custom_parameters(temp_dir, sample_data):
    """Test plotting with custom parameters."""
    # Test with custom figure size
    output_file = temp_dir / 'custom_size.png'
    plot_rmsd_distribution(
        sample_data['rmsd_values'],
        output_file,
        figsize=(10, 6),
        title='Custom RMSD Distribution'
    )
    assert output_file.exists()
    
    # Test with custom color scheme
    output_file = temp_dir / 'custom_colors.png'
    plot_contact_map(
        sample_data['contact_map'],
        output_file,
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    assert output_file.exists()
    
    # Test with custom labels
    output_file = temp_dir / 'custom_labels.png'
    plot_tm_score_distribution(
        sample_data['tm_scores'],
        output_file,
        xlabel='TM-Score',
        ylabel='Count'
    )
    assert output_file.exists()

def test_plot_with_save_options(temp_dir, sample_data):
    """Test plotting with different save options."""
    # Test with different formats
    for fmt in ['png', 'jpg', 'pdf', 'svg']:
        output_file = temp_dir / f'test.{fmt}'
        plot_rmsd_distribution(sample_data['rmsd_values'], output_file)
        assert output_file.exists()
    
    # Test with different DPI
    output_file = temp_dir / 'high_dpi.png'
    plot_rmsd_distribution(sample_data['rmsd_values'], output_file, dpi=300)
    assert output_file.exists()
    
    # Test with transparent background
    output_file = temp_dir / 'transparent.png'
    plot_rmsd_distribution(sample_data['rmsd_values'], output_file, transparent=True)
    assert output_file.exists()