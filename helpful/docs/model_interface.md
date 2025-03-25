# Model Interface

The ModelInterface class provides a unified interface for working with different protein structure prediction models, enabling seamless integration and switching between various model architectures.

## Key Features

### 1. Unified Model Management

- Standardized prediction interface across all supported models
- Automatic model loading and initialization
- Dynamic model switching
- Configurable model parameters
- Memory-efficient operation

### 2. Supported Models

- ESM (Meta AI)
- AlphaFold2 (DeepMind)
- RoseTTAFold
- ColabFold
- OpenFold
- Custom model support

### 3. Memory Management

- Automatic model weight caching
- GPU memory optimization
- Gradient checkpointing support
- Memory-efficient inference

### 4. Output Formats

- PDB format
- mmCIF format
- Internal structure representation
- Confidence metrics
- Per-residue predictions

## Configuration

### Basic Configuration

```python
from src.interfaces import ModelInterface, BaseModelConfig

config = BaseModelConfig(
    name="esm2_t33_650M_UR50D",
    model_type="esm",
    output_format="pdb",
    device="cuda:0",
    precision="float32"
)
```

### Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| name | str | Model name/version | Required |
| model_type | str | Model architecture type | Required |
| output_format | str | Output format (pdb/mmcif) | "pdb" |
| device | str | Computing device | "cuda:0" |
| precision | str | Computation precision | "float32" |
| batch_size | int | Batch size for predictions | 1 |
| max_seq_len | int | Maximum sequence length | None |
| cache_dir | str | Directory for model caching | None |

## Usage Examples

### Basic Structure Prediction

```python
from src.interfaces import ModelInterface, BaseModelConfig

# Initialize interface
config = BaseModelConfig(
    name="esm2_t33_650M_UR50D",
    model_type="esm"
)
model = ModelInterface(config)

# Single sequence prediction
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
result = model.predict_structure(sequence)

print(f"Output PDB file: {result['structure']}")
print(f"Confidence score: {result['confidence'].mean():.3f}")
```

### Batch Prediction

```python
# Multiple sequence prediction
sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
]

results = model.predict_structures(
    sequences,
    batch_size=2,
    show_progress=True
)

for seq_id, result in results.items():
    print(f"\nResults for sequence {seq_id}:")
    print(f"Structure file: {result['structure']}")
    print(f"Confidence: {result['confidence'].mean():.3f}")
```

### Advanced Configuration

```python
config = BaseModelConfig(
    name="esm2_t33_650M_UR50D",
    model_type="esm",
    output_format="mmcif",
    device="cuda:0",
    precision="mixed",
    batch_size=4,
    max_seq_len=1000,
    cache_dir="./model_cache",
    extra_options={
        "use_templates": True,
        "num_recycles": 3,
        "compute_embeddings": True
    }
)
```

## Error Handling

The interface provides comprehensive error handling for common issues:

1. Model Loading Errors

```python
try:
    model = ModelInterface(config)
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
    # Handle error or try alternative model
```

1. Prediction Errors

```python
try:
    result = model.predict_structure(sequence)
except PredictionError as e:
    print(f"Prediction failed: {e}")
    # Implement fallback or retry logic
```

1. Memory Errors

```python
try:
    results = model.predict_structures(large_batch)
except MemoryError as e:
    print(f"Out of memory: {e}")
    # Reduce batch size or free memory
```

## Best Practices

1. Model Selection

- Choose appropriate model for sequence length
- Consider memory requirements
- Balance speed vs accuracy

1. Memory Management

- Use appropriate precision (float32/mixed/float16)
- Enable gradient checkpointing for long sequences
- Implement proper cleanup after predictions

1. Batch Processing

- Use appropriate batch sizes
- Enable progress tracking for large batches
- Implement error recovery

## Performance Optimization

### 1. Memory Optimization

```python
model.enable_memory_optimization(
    use_gradient_checkpointing=True,
    clear_cache_between_batches=True,
    min_free_memory=0.2
)
```

### 2. Batch Size Optimization

```python
model.optimize_batch_size(
    target_memory_usage=0.8,
    min_batch_size=1,
    max_batch_size=32
)
```

### 3. Multi-GPU Support

```python
config = BaseModelConfig(
    name="esm2_t33_650M_UR50D",
    model_type="esm",
    devices=["cuda:0", "cuda:1"],
    strategy="data_parallel"
)
```

## Integration with Other Components

### 1. Ensemble Prediction

```python
from src.core.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor([
    ModelInterface(config1),
    ModelInterface(config2)
])
result = ensemble.predict(sequence)
```

### 2. Distributed Inference

```python
from src.core.distributed import DistributedInferenceManager

dist_manager = DistributedInferenceManager(dist_config)
dist_manager.register_model(model)
results = dist_manager.run_inference(sequences)
```

### 3. Database Integration

```python
from src.core.database import DatabaseManager

db = DatabaseManager("predictions.db")
result = model.predict_structure(sequence)
db.store_prediction(result)
```

## Monitoring and Metrics

The interface provides various monitoring capabilities:

1. Performance Metrics

- Prediction time
- Memory usage
- GPU utilization
- Cache hit rates

1. Quality Metrics

- Confidence scores
- Structure validation
- Model-specific metrics

1. Resource Usage

- Memory tracking
- GPU memory allocation
- Cache statistics

## Troubleshooting

Common issues and solutions:

1. Model Loading Issues

- Verify model availability
- Check cache directory permissions
- Validate model configuration

1. Memory Problems

- Reduce batch size
- Enable memory optimizations
- Clear cache between predictions

1. Performance Issues

- Profile prediction pipeline
- Optimize batch size
- Check GPU utilization
