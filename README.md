# Proteus: Multi-Model Protein Structure Analysis

[![PyPI version](https://badge.fury.io/py/proteus-multi-model-analysis.svg)](https://badge.fury.io/py/proteus-multi-model-analysis)
[![Python Version](https://img.shields.io/pypi/pyversions/proteus-multi-model-analysis.svg)](https://pypi.org/project/proteus-multi-model-analysis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/proteus-multi-model-analysis/badge/?version=latest)](https://proteus-multi-model-analysis.readthedocs.io/en/latest/?badge=latest)

A comprehensive framework for protein structure prediction and analysis using multiple state-of-the-art models.

## 🚀 Features

- **Multiple Model Integration**
  - AlphaFold2
  - ESM
  - RoseTTAFold
  - OpenFold
  - ColabFold

- **Advanced Capabilities**
  - Unified prediction interface
  - Ensemble predictions
  - Structure comparison & visualization
  - Distributed inference
  - GPU memory optimization
  - Model versioning & checkpointing

## 📋 Requirements

- Python ≥ 3.8
- CUDA-compatible GPU (optional, for GPU acceleration)
- 8GB RAM minimum (16GB+ recommended)

## 🔧 Installation

### Basic Installation

```bash
pip install proteus-multi-model-analysis
```

### With GPU Support

```bash
pip install "proteus-multi-model-analysis[gpu]"
```

### Development Installation

```bash
pip install "proteus-multi-model-analysis[dev]"
```

## 🎯 Quick Start

```python
from proteus import ProteinAnalyzer
from proteus.models import ESMPredictor

# Initialize analyzer with a model
analyzer = ProteinAnalyzer(model=ESMPredictor())

# Predict structure for a sequence
sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
prediction = analyzer.predict_structure(sequence)

# Visualize the prediction
analyzer.visualize(prediction)
```

## 📚 Documentation

Full documentation is available at [proteus-multi-model-analysis.readthedocs.io](https://proteus-multi-model-analysis.readthedocs.io/)

Key documentation sections:

- [API Reference](docs/api_integration.md)
- [Model Interfaces](docs/model_interface.md)
- [Ensemble Methods](docs/ensemble.md)
- [Distributed Computing](docs/distributed_inference.md)
- [Visualization Tools](docs/visualization.md)

## 🧪 Testing

```bash
# Run the test suite
pytest

# Run with coverage report
pytest --cov=proteus tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use Proteus in your research, please cite:

```bibtex
@software{proteus2025,
  author = {Frazier, Jack},
  title = {Proteus: Multi-Model Protein Structure Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JackFrazier-jnf44/proteus}
}
```

## 📬 Contact

- School Email - [jnf44@cornell.edu](mailto:jnf44@cornell.edu)

- Personal Email - [jack.frazier03@gmail.com](mailto:jack.frazier03@gmail.com)

- Project Link: [https://github.com/JackFrazier-jnf44/proteus](https://github.com/JackFrazier-jnf44/proteus)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Proteus is a comprehensive framework for analyzing protein structures using multiple state-of-the-art deep learning models. It provides a unified interface for structure prediction, embedding extraction, and comparative analysis across different models like AlphaFold2, ESM, OpenFold, RosettaFold, and ColabFold.

## Key Features

### Supported Models

- **AlphaFold2**: State-of-the-art protein structure prediction
- **ESM**: Evolutionary scale modeling and embeddings
- **OpenFold**: Open-source implementation of AlphaFold2
- **RosettaFold**: Fast and accurate structure prediction
- **ColabFold**: Efficient protein structure prediction

### Core Capabilities

- **Unified Interface**: Common API across all supported models
- **Batch Processing**: Efficient processing of multiple sequences ([docs](helpful/docs/batch_processing.md))
- **Distributed Computing**: Scale across multiple machines ([docs](helpful/docs/distributed_inference.md))
- **Model Versioning**: Track and manage model versions ([docs](helpful/docs/model_versioning.md))
- **API Integration**: Easy integration with external services ([docs](helpful/docs/api_integration.md))

### Analysis Tools

- **Structure Comparison**: Compare predicted structures
- **Statistical Analysis**: Comprehensive statistical methods
- **Visualization**: Advanced structure visualization tools
- **Ensemble Methods**: Combine predictions from multiple models
- **Quantization**: Model compression and optimization

## Installation

```bash
# Using pip
pip install proteus

# Using poetry (recommended)
poetry add proteus

# For GPU support
poetry add proteus[gpu]
```

## Quickstart

```python
from src.interfaces import ModelInterface
from src.core.batch import BatchProcessor

# Initialize model
model = ModelInterface(model_type="alphafold2")

# Process a sequence
sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
result = model.predict_structure(sequence)

# Batch processing
processor = BatchProcessor(max_batch_size=10)
sequences = [sequence1, sequence2, sequence3]
results = processor.process_batch(model, sequences)
```

For more detailed examples, check out:

- [Basic Usage](helpful/docs/model_interface.md)
- [Batch Processing](helpful/docs/batch_processing.md)
- [Advanced Features](helpful/docs/ensemble.md)

## Project Structure

```bibtex
proteus/
├── src/                    # Core source code
│   ├── interfaces/         # Model interfaces
│   ├── core/              # Core functionality
│   │   ├── batch/         # Batch processing
│   │   ├── distributed/   # Distributed computing
│   │   └── visualization/ # Visualization tools
│   └── analysis/          # Analysis tools
├── helpful/
│   ├── docs/              # Documentation
│   └── examples/          # Example scripts
└── tests/                 # Test suite
```

## Documentation

Comprehensive documentation is available in the [docs](helpful/docs/) directory:

- [Model Interface Guide](helpful/docs/model_interface.md)
- [Batch Processing Guide](helpful/docs/batch_processing.md)
- [Distributed Computing](helpful/docs/distributed_inference.md)
- [API Integration](helpful/docs/api_integration.md)
- [Database Management](helpful/docs/database_management.md)
- [Model Versioning](helpful/docs/model_versioning.md)
- [Testing Guide](helpful/docs/TESTING.md)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended)
- See [requirements.txt](requirements.txt) for full list

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Proteus in your research, please cite:

```bibtex
@software{proteus2025,
  author = {Frazier, Jack},
  title = {Proteus: Multi-Model Protein Structure Analysis Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JackFrazier-jnf44/proteus}
}
```

## Contact

- Jack Frazier
- School Email: <jnf44@cornell.edu>
- Personal Email: <jack.frazier03@gmail.com>
- Project Link: <https://github.com/JackFrazier-jnf44/proteus>
