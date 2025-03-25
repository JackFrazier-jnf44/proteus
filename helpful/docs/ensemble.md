# Ensemble Prediction

The ensemble prediction system enables combining predictions from multiple protein structure prediction models to achieve more robust and accurate results.

## Key Features

### 1. Ensemble Methods

#### Weighted Average

Combines predictions using fixed weights for each model:

```python
config = EnsembleConfig(
    method=EnsembleMethod.WEIGHTED_AVERAGE,
    weights={"esm": 0.6, "alphafold": 0.4}
)
```

#### Confidence-Weighted

Weights predictions based on model confidence scores:

```python
config = EnsembleConfig(
    method=EnsembleMethod.CONFIDENCE_WEIGHTED,
    confidence_threshold=0.7
)
```

#### Voting

Uses consensus-based approach for structure prediction:

```python
config = EnsembleConfig(
    method=EnsembleMethod.VOTING,
    voting_threshold=0.5
)
```

#### Dynamic Weighting

Adapts weights based on prediction quality:

```python
config = EnsembleConfig(
    method=EnsembleMethod.DYNAMIC_WEIGHTED,
    initial_weights={"model1": 0.5, "model2": 0.5},
    adaptation_rate=0.1
)
```

### 2. Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| method | EnsembleMethod | Ensemble combination method | WEIGHTED_AVERAGE |
| weights | Dict[str, float] | Model-specific weights | None |
| confidence_threshold | float | Minimum confidence for inclusion | 0.5 |
| voting_threshold | float | Consensus threshold for voting | 0.5 |
| adaptation_rate | float | Weight adaptation rate | 0.1 |
| device | str | Computing device | "cpu" |

## Usage Examples

### Basic Ensemble Prediction

```python
from src.core.ensemble import EnsemblePredictor, EnsembleConfig, EnsembleMethod
from src.interfaces import ModelInterface, BaseModelConfig

# Configure models
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

# Configure ensemble
config = EnsembleConfig(
    method=EnsembleMethod.WEIGHTED_AVERAGE,
    weights={"esm": 0.6, "alphafold": 0.4}
)
ensemble = EnsemblePredictor(config)

# Make predictions
sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
predictions = {
    "esm": models[0].predict_structure(sequence),
    "alphafold": models[1].predict_structure(sequence)
}

# Combine predictions
result = ensemble.combine_predictions(predictions)
```

### Confidence-Weighted Ensemble

```python
# Configure ensemble with confidence weighting
config = EnsembleConfig(
    method=EnsembleMethod.CONFIDENCE_WEIGHTED,
    confidence_threshold=0.7
)
ensemble = EnsemblePredictor(config)

# Process multiple sequences
sequences = [
    "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
]

for seq in sequences:
    predictions = {
        f"model_{i}": model.predict_structure(seq)
        for i, model in enumerate(models)
    }
    result = ensemble.combine_predictions(predictions)
```

### Dynamic Weight Adaptation

```python
# Configure ensemble with dynamic weighting
config = EnsembleConfig(
    method=EnsembleMethod.DYNAMIC_WEIGHTED,
    initial_weights={"esm": 0.5, "alphafold": 0.5},
    adaptation_rate=0.1
)
ensemble = EnsemblePredictor(config)

# Train ensemble weights using known structures
training_sequences = [
    ("SEQUENCE1", "PDB_ID1"),
    ("SEQUENCE2", "PDB_ID2")
]

for seq, pdb_id in training_sequences:
    predictions = get_model_predictions(seq)
    ensemble.update_weights(predictions, reference_pdb=pdb_id)
```

## Integration with Other Components

### 1. Visualization

```python
from src.core.visualization import plot_ensemble_comparison

plot_ensemble_comparison(
    predictions=predictions,
    result=result,
    output_file="ensemble_comparison.png",
    plot_config={
        "figure_size": (15, 10),
        "dpi": 300,
        "show_confidence": True,
        "show_rmsd": True
    }
)
```

### 2. Database Storage

```python
from src.core.database import DatabaseManager

db = DatabaseManager("predictions.db")
db.store_prediction(result, metadata={
    "sequence": sequence,
    "method": "weighted_ensemble",
    "models_used": len(predictions)
})
```

### 3. Distributed Processing

```python
from src.core.distributed import DistributedInferenceManager

dist_manager = DistributedInferenceManager(dist_config)
ensemble_predictor = EnsemblePredictor(ensemble_config)

# Run distributed ensemble prediction
results = dist_manager.run_ensemble_inference(
    ensemble_predictor,
    sequences,
    models
)
```

## Best Practices

1. Model Selection

- Choose complementary models
- Consider computational cost
- Balance accuracy vs. speed

2. Weight Configuration

- Start with equal weights
- Adjust based on model performance
- Use confidence scores when available

3. Performance Optimization

- Enable GPU acceleration
- Use batch processing
- Implement caching

## Error Handling

The ensemble system handles several types of errors:

1. Prediction Errors

```python
try:
    result = ensemble.combine_predictions(predictions)
except EnsembleError as e:
    print(f"Ensemble prediction failed: {e}")
    # Implement fallback strategy
```

2. Weight Validation

```python
try:
    config = EnsembleConfig(
        method=EnsembleMethod.WEIGHTED_AVERAGE,
        weights={"model1": 0.6, "model2": 0.5}  # Sum > 1
    )
except ValueError as e:
    print(f"Invalid weights: {e}")
```

3. Missing Predictions

```python
try:
    result = ensemble.combine_predictions(incomplete_predictions)
except EnsembleError as e:
    print(f"Missing predictions: {e}")
    # Request missing predictions
```

## Performance Considerations

### 1. Memory Management

```python
ensemble.enable_memory_optimization(
    cache_predictions=True,
    clear_cache_between_batches=True
)
```

### 2. GPU Acceleration

```python
config = EnsembleConfig(
    method=EnsembleMethod.WEIGHTED_AVERAGE,
    weights=weights,
    device="cuda:0"
)
```

### 3. Batch Processing

```python
results = ensemble.predict_batch(
    sequences,
    batch_size=4,
    show_progress=True
)
```

## Monitoring and Metrics

The ensemble system provides various monitoring capabilities:

1. Prediction Quality

- Per-model confidence scores
- Ensemble agreement metrics
- Structure quality assessment

2. Performance Metrics

- Prediction time
- Memory usage
- GPU utilization

3. Weight Evolution

- Weight adaptation history
- Model contribution analysis
- Convergence monitoring

## Troubleshooting

Common issues and solutions:

1. Memory Issues

- Reduce batch size
- Enable memory optimization
- Clear cache regularly

2. Performance Problems

- Use GPU acceleration
- Optimize batch size
- Monitor resource usage

3. Accuracy Issues

- Validate model weights
- Check confidence thresholds
- Analyze model contributions
