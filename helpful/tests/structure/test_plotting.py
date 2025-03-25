import pytest
import numpy as np
import matplotlib.pyplot as plt
from multi_model_analysis.utils.plotting import generate_plots

def test_generate_plots():
    # Create dummy data for testing
    data = {
        'x': np.linspace(0, 10, 100),
        'y1': np.sin(np.linspace(0, 10, 100)),
        'y2': np.cos(np.linspace(0, 10, 100)),
    }
    
    # Call the function to generate plots
    fig = generate_plots(data)
    
    # Check if the figure is created
    assert isinstance(fig, plt.Figure)
    
    # Check if the number of subplots is as expected
    assert len(fig.get_axes()) == 2  # Assuming we expect 2 subplots

    # Check if the data in the first subplot is correct
    ax1 = fig.get_axes()[0]
    assert ax1.lines[0].get_ydata().tolist() == pytest.approx(data['y1'].tolist(), rel=1e-2)

    # Check if the data in the second subplot is correct
    ax2 = fig.get_axes()[1]
    assert ax2.lines[0].get_ydata().tolist() == pytest.approx(data['y2'].tolist(), rel=1e-2)

    plt.close(fig)  # Close the figure after testing to avoid display issues

def test_generate_plots_empty_data():
    # Test with empty data
    data = {}
    
    with pytest.raises(ValueError):
        generate_plots(data)  # Expecting a ValueError for empty data input

def test_generate_plots_invalid_data():
    # Test with invalid data format
    data = "invalid_data"
    
    with pytest.raises(TypeError):
        generate_plots(data)  # Expecting a TypeError for invalid data input