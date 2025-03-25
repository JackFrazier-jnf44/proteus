# Batch Processing

## Overview

The batch processing system in Proteus enables efficient processing of multiple protein sequences using various models. It provides parallel processing capabilities, progress tracking, and error handling for large-scale structure prediction tasks.

## Key Features

## Batch Processing

- Parallel sequence processing
- Memory-efficient batching
- Progress tracking
- Error handling and recovery
- Result aggregation

### Resource Management

- CPU/GPU utilization control
- Memory usage monitoring
- Batch size optimization
- Queue management
- Resource allocation

## Usage

### Basic Batch Processing

```python
from src.core.batch import BatchProcessor
from src.interfaces import ModelInterface

# Initialize processor
processor = BatchProcessor(
    max_batch_size=10,
    num_workers=4
)

# Create model
model = ModelInterface(name="esm2_t33_650M_UR50D")

# Process sequences
sequences = [
    "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    # ... more sequences
]

results = processor.process_batch(
    model=model,
    sequences=sequences,
    prediction_type="structure"
)
```

### Advanced Batch Configuration

```python
from src.core.batch import BatchConfig

# Configure batch processing
config = BatchConfig(
    max_batch_size=10,
    num_workers=4,
    gpu_ids=[0, 1],
    memory_limit_gb=32,
    timeout_per_sequence=300,
    retry_attempts=3
)

processor = BatchProcessor(config)
```

## Configuration Options

### Batch Processor Configuration

| Option | Description | Default |
|--------|-------------|---------|
| max_batch_size | Maximum sequences per batch | 10 |
| num_workers | Number of parallel workers | CPU count |
| gpu_ids | List of GPU IDs to use | None |
| memory_limit_gb | Maximum memory usage | Available RAM |
| timeout_per_sequence | Timeout in seconds | 300 |

### Resource Configuration

| Option | Description | Default |
|--------|-------------|---------|
| cpu_threads | Threads per worker | 1 |
| gpu_memory_fraction | GPU memory fraction | 0.9 |
| prefetch_batches | Number of batches to prefetch | 2 |
| cleanup_interval | Cleanup interval in seconds | 60 |

## Best Practices

### Batch Size Selection

1. Consider model memory requirements
2. Account for available GPU memory
3. Balance throughput and latency
4. Monitor resource utilization
5. Adjust based on sequence lengths

### Resource Management

1. Monitor memory usage
2. Use appropriate number of workers
3. Enable GPU memory optimization
4. Implement proper error handling
5. Regular resource cleanup

## Error Handling

### Batch Processing Errors

```python
try:
    results = processor.process_batch(model, sequences)
except BatchProcessingError as e:
    # Handle batch processing error
    failed_sequences = e.failed_sequences
    error_details = e.error_details
except ResourceExhaustedError:
    # Handle resource exhaustion
except TimeoutError:
    # Handle timeout
```

### Recovery Strategies

```python
# Configure recovery options
processor = BatchProcessor(
    recovery_strategy="continue",  # or "restart"
    checkpoint_interval=10
)

# Process with recovery
results = processor.process_batch_with_recovery(
    model=model,
    sequences=sequences,
    checkpoint_file="checkpoint.pkl"
)
```

## Performance Optimization

### Memory Optimization

- Batch size tuning
- Memory cleanup
- Resource monitoring
- Caching strategies

### Processing Optimization

- Worker pool management
- GPU memory optimization
- Load balancing
- Pipeline optimization

## Integration

### With Model Interface

```python
from src.interfaces import ModelInterface
from src.core.batch import BatchProcessor

model = ModelInterface(
    name="esm2_t33_650M_UR50D",
    batch_processor=BatchProcessor()
)
```

### With Distributed Systems

```python
from src.core.batch import DistributedBatchProcessor

dist_processor = DistributedBatchProcessor(
    scheduler="dask",
    num_workers=4,
    worker_resources={
        "memory": "8GB",
        "gpu": 1
    }
)
```

## Monitoring and Metrics

### Batch Metrics

- Processing throughput
- Success/failure rates
- Resource utilization
- Processing times

### Resource Metrics

- Memory usage
- GPU utilization
- Worker status
- Queue lengths

## Troubleshooting

### Common Issues

1. Memory exhaustion
2. GPU out of memory
3. Worker crashes
4. Timeout errors

### Performance Issues

1. Slow processing
2. High memory usage
3. GPU underutilization
4. Worker bottlenecks

## API Reference

### Batch Processor API

- `process_batch(model, sequences, **kwargs)`
- `process_batch_with_recovery(model, sequences, checkpoint_file)`
- `get_batch_status(batch_id)`
- `cancel_batch(batch_id)`
- `cleanup_resources()`

### Resource Manager API

- `allocate_resources(requirements)`
- `release_resources(resource_id)`
- `get_resource_status()`
- `optimize_resource_usage()`
- `cleanup_resources()`

## Examples

### Basic Usage

```python
from src.core.batch import BatchProcessor
from src.interfaces import ModelInterface

# Initialize
processor = BatchProcessor(max_batch_size=10)
model = ModelInterface(name="esm2_t33_650M_UR50D")

# Process sequences
sequences = ["SEQ1", "SEQ2", "SEQ3"]
results = processor.process_batch(model, sequences)

# Get results
for seq_id, result in results.items():
    print(f"Sequence {seq_id}: {result['confidence']}")
```

### Advanced Usage

```python
from src.core.batch import BatchProcessor, BatchConfig

# Configure processor
config = BatchConfig(
    max_batch_size=10,
    num_workers=4,
    gpu_ids=[0, 1],
    memory_limit_gb=32
)

processor = BatchProcessor(config)

# Process with monitoring
results = processor.process_batch(
    model=model,
    sequences=sequences,
    callbacks=[
        lambda batch_id, status: print(f"Batch {batch_id}: {status}")
    ]
)
```
