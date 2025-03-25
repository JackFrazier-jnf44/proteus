from pathlib import Path
from typing import Dict, Any

# Default paths
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OPENFOLD_DIR = DEFAULT_MODEL_DIR / "openfold"
DEFAULT_ESM_DIR = DEFAULT_MODEL_DIR / "esm"
DEFAULT_ALPHAFOLD_DIR = DEFAULT_MODEL_DIR / "alphafold"
DEFAULT_ROSETTAFOLD_DIR = DEFAULT_MODEL_DIR / "rosettafold"

# Model configurations
DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "esm": {
        "model_name": "esm2_t36_3B_UR50D",
        "embedding_config": {
            "last_hidden": {"dimension": 1280},
            "pooled": {"dimension": 1280}
        }
    },
    "openfold": {
        "model_path": str(DEFAULT_OPENFOLD_DIR / "weights"),
        "config_path": str(DEFAULT_OPENFOLD_DIR / "config.yaml"),
        "embedding_config": {
            "structure_embedding": {"dimension": 512},
            "pair_embedding": {"dimension": 256}
        }
    },
    "alphafold": {
        "model_path": str(DEFAULT_ALPHAFOLD_DIR / "weights"),
        "config_path": str(DEFAULT_ALPHAFOLD_DIR / "config.yaml"),
        "embedding_config": {
            "structure_embedding": {"dimension": 1024},
            "pair_embedding": {"dimension": 512},
            "msa_embedding": {"dimension": 256}
        }
    },
    "rosettafold": {
        "model_path": str(DEFAULT_ROSETTAFOLD_DIR / "weights"),
        "config_path": str(DEFAULT_ROSETTAFOLD_DIR / "config.yaml"),
        "embedding_config": {
            "structure_embedding": {"dimension": 768},
            "pair_embedding": {"dimension": 384},
            "msa_embedding": {"dimension": 192}
        }
    }
}

# Visualization settings
PLOT_SETTINGS = {
    "figure_size": (15, 10),
    "dpi": 300,
    "output_format": "pdf"
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "distance_threshold": 8.0,  # Å
    "confidence_threshold": 0.7,
    "embedding_reduction_dim": 2
}

# File formats
SUPPORTED_FORMATS = {
    "structure": [".pdb", ".mmcif"],
    "output": [".pdf", ".png"]
}

# Error messages
ERROR_MESSAGES = {
    "model_not_found": "Model {model_name} not found in {model_dir}",
    "invalid_input": "Invalid input sequence: {sequence}",
    "gpu_memory": "Insufficient GPU memory for model {model_name}",
    "file_not_found": "File not found: {file_path}"
}

# Add to configuration:
API_SETTINGS = {
    "rate_limits": {},
    "retry_settings": {},
    "timeout_settings": {}
}

DISTRIBUTED_SETTINGS = {
    "node_configuration": {},
    "synchronization_settings": {},
    "fallback_strategies": {}
} 