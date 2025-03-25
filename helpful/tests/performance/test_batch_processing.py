"""Tests for batch processing functionality."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from proteus.core.batch import (
    BatchProcessor,
    BatchConfig,
    DistributedBatchProcessor,
    BatchProcessingError,
    ResourceExhaustedError
)
from proteus.interfaces import ModelInterface, BaseModelConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def batch_processor():
    """Create a batch processor instance."""
    return BatchProcessor(
        max_batch_size=5,
        num_workers=2
    )

@pytest.fixture
def sample_model():
    """Create a sample model instance."""
    return ModelInterface(
        BaseModelConfig(
            name="test_model",
            model_type="esm"
        )
    )

@pytest.fixture
def sample_sequences():
    """Create sample sequences for testing."""
    return [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
    ]

def test_basic_batch_processing(batch_processor, sample_model, sample_sequences):
    """Test basic batch processing functionality."""
    # Process sequences
    results = batch_processor.process_batch(
        model=sample_model,
        sequences=sample_sequences,
        prediction_type="structure"
    )
    
    # Check results
    assert len(results) == len(sample_sequences)
    for result in results:
        assert 'structure' in result
        assert 'confidence' in result
        assert 'processing_time' in result
        assert result['confidence'] >= 0 and result['confidence'] <= 1

def test_batch_configuration():
    """Test batch processor configuration."""
    # Test valid configuration
    config = BatchConfig(
        max_batch_size=10,
        num_workers=4,
        gpu_ids=[0],
        memory_limit_gb=32,
        timeout_per_sequence=300
    )
    processor = BatchProcessor(config)
    assert processor.config.max_batch_size == 10
    assert processor.config.num_workers == 4
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        BatchConfig(max_batch_size=0)
    
    with pytest.raises(ValueError):
        BatchConfig(num_workers=-1)
    
    with pytest.raises(ValueError):
        BatchConfig(memory_limit_gb=0)

def test_batch_callbacks(batch_processor, sample_model, sample_sequences):
    """Test batch processing callbacks."""
    # Initialize callback tracking
    callback_data = {
        'batch_started': 0,
        'batch_completed': 0,
        'sequences_processed': 0
    }
    
    def batch_callback(batch_id, status):
        if status == 'started':
            callback_data['batch_started'] += 1
        elif status == 'completed':
            callback_data['batch_completed'] += 1
            callback_data['sequences_processed'] += len(batch_id)
    
    # Process with callback
    results = batch_processor.process_batch(
        model=sample_model,
        sequences=sample_sequences,
        callbacks=[batch_callback]
    )
    
    # Check callback data
    assert callback_data['batch_started'] > 0
    assert callback_data['batch_completed'] > 0
    assert callback_data['sequences_processed'] == len(sample_sequences)

def test_batch_error_handling(batch_processor, sample_model):
    """Test batch processing error handling."""
    # Test with invalid sequences
    invalid_sequences = ["INVALID", "NOT_AA_SEQUENCE", "123"]
    
    with pytest.raises(BatchProcessingError) as exc_info:
        batch_processor.process_batch(
            model=sample_model,
            sequences=invalid_sequences
        )
    
    assert len(exc_info.value.failed_sequences) > 0
    assert exc_info.value.error_details is not None

def test_batch_recovery(batch_processor, sample_model, sample_sequences, temp_dir):
    """Test batch processing recovery functionality."""
    # Configure recovery
    checkpoint_file = temp_dir / "checkpoint.pkl"
    processor = BatchProcessor(
        recovery_strategy="continue",
        checkpoint_interval=1
    )
    
    try:
        # Process with recovery
        results = processor.process_batch_with_recovery(
            model=sample_model,
            sequences=sample_sequences,
            checkpoint_file=checkpoint_file
        )
        
        assert len(results) == len(sample_sequences)
        
    except Exception:
        # Test recovery from checkpoint
        results = processor.recover_from_checkpoint(checkpoint_file)
        assert len(results) > 0

def test_resource_management(batch_processor, sample_model, sample_sequences):
    """Test resource management during batch processing."""
    # Configure resource limits
    processor = BatchProcessor(
        max_batch_size=2,
        num_workers=1,
        memory_limit_gb=1
    )
    
    # Monitor resource usage
    resource_usage = []
    def resource_callback(batch_id, status, metrics):
        if status == 'processing':
            resource_usage.append(metrics['memory_usage_gb'])
    
    # Process with resource monitoring
    try:
        processor.process_batch(
            model=sample_model,
            sequences=sample_sequences,
            callbacks=[resource_callback]
        )
    except ResourceExhaustedError:
        assert max(resource_usage) >= 1.0

def test_distributed_batch_processing():
    """Test distributed batch processing functionality."""
    try:
        # Initialize distributed processor
        processor = DistributedBatchProcessor(
            scheduler="local",
            num_workers=2
        )
        
        # Create model and sequences
        model = ModelInterface(
            BaseModelConfig(
                name="test_model",
                model_type="esm"
            )
        )
        
        sequences = [
            "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(50)])
            for _ in range(3)
        ]
        
        # Process sequences
        results = processor.process_batch(
            model=model,
            sequences=sequences
        )
        
        assert len(results) == len(sequences)
        for result in results:
            assert 'worker_id' in result
        
    except ImportError:
        pytest.skip("Distributed backend not available")

def test_batch_size_optimization(batch_processor, sample_model):
    """Test batch size optimization."""
    # Generate sequences of different lengths
    sequences = [
        "".join(["A" for _ in range(length)])
        for length in [50, 100, 200, 400]
    ]
    
    # Process with auto batch size
    processor = BatchProcessor(
        max_batch_size="auto",
        optimize_batch_size=True
    )
    
    results = processor.process_batch(
        model=sample_model,
        sequences=sequences
    )
    
    assert len(results) == len(sequences)
    assert processor.get_optimal_batch_size() > 0

def test_batch_priority_queue(batch_processor, sample_model):
    """Test batch priority queue functionality."""
    # Create sequences with priorities
    sequences = [
        ("SEQUENCE1", 1),
        ("SEQUENCE2", 3),
        ("SEQUENCE3", 2)
    ]
    
    # Process with priorities
    results = batch_processor.process_batch_with_priority(
        model=sample_model,
        sequences=[seq for seq, _ in sequences],
        priorities=[pri for _, pri in sequences]
    )
    
    assert len(results) == len(sequences)
    
    # Check processing order from logs
    processing_order = batch_processor.get_processing_order()
    assert processing_order[0] == 1  # Highest priority first

def test_batch_progress_tracking(batch_processor, sample_model, sample_sequences):
    """Test batch progress tracking functionality."""
    # Initialize progress tracking
    progress_data = {
        'total_progress': 0,
        'current_batch': 0
    }
    
    def progress_callback(progress, batch_num, total_batches):
        progress_data['total_progress'] = progress
        progress_data['current_batch'] = batch_num
    
    # Process with progress tracking
    results = batch_processor.process_batch(
        model=sample_model,
        sequences=sample_sequences,
        progress_callback=progress_callback
    )
    
    assert progress_data['total_progress'] == 100
    assert progress_data['current_batch'] > 0 