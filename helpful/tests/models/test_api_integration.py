"""Tests for API integration functionality."""

import pytest
import responses
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from proteus.interfaces import (
    APIManager,
    APIConfig,
    ModelInterface,
    BaseModelConfig,
    APIError,
    AuthenticationError,
    RateLimitError
)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def api_manager():
    """Create an API manager instance."""
    return APIManager(
        service="alphafold",
        api_key="test_key"
    )

@pytest.fixture
def mock_api_response():
    """Create mock API response data."""
    return {
        "prediction_id": "test_pred_123",
        "confidence": 0.95,
        "processing_time": 10.5,
        "structure": {
            "atoms": [],
            "bonds": []
        }
    }

@responses.activate
def test_basic_api_integration(api_manager, mock_api_response):
    """Test basic API integration functionality."""
    # Mock API endpoint
    responses.add(
        responses.POST,
        "https://api.example.com/predict",
        json=mock_api_response,
        status=200
    )
    
    # Test prediction request
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    result = api_manager.predict_structure(sequence)
    
    # Verify response
    assert result["prediction_id"] == mock_api_response["prediction_id"]
    assert result["confidence"] == mock_api_response["confidence"]
    assert "structure" in result

def test_api_configuration():
    """Test API configuration."""
    # Test valid configuration
    config = APIConfig(
        service="alphafold",
        api_key="test_key",
        base_url="https://api.example.com",
        max_retries=3,
        timeout=300
    )
    manager = APIManager(config)
    assert manager.config.service == "alphafold"
    assert manager.config.max_retries == 3
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        APIConfig(service="invalid")
    
    with pytest.raises(ValueError):
        APIConfig(service="alphafold", max_retries=-1)

@responses.activate
def test_api_error_handling(api_manager):
    """Test API error handling."""
    # Mock API errors
    responses.add(
        responses.POST,
        "https://api.example.com/predict",
        json={"error": "Authentication failed"},
        status=401
    )
    
    responses.add(
        responses.POST,
        "https://api.example.com/predict",
        json={"error": "Rate limit exceeded"},
        status=429
    )
    
    # Test authentication error
    with pytest.raises(AuthenticationError):
        api_manager.predict_structure("SEQUENCE")
    
    # Test rate limit error
    with pytest.raises(RateLimitError):
        api_manager.predict_structure("SEQUENCE")

@responses.activate
def test_retry_handling(api_manager, mock_api_response):
    """Test retry handling functionality."""
    # Mock failed attempts followed by success
    responses.add(
        responses.POST,
        "https://api.example.com/predict",
        json={"error": "Server error"},
        status=500
    )
    
    responses.add(
        responses.POST,
        "https://api.example.com/predict",
        json=mock_api_response,
        status=200
    )
    
    # Configure retry
    api_manager.config.max_retries = 2
    api_manager.config.retry_strategy = "exponential"
    
    # Test prediction with retry
    result = api_manager.predict_structure_with_retry(
        sequence="SEQUENCE",
        backoff_factor=1.5
    )
    
    assert result["prediction_id"] == mock_api_response["prediction_id"]
    assert result["retry_count"] == 1

def test_rate_limiting(api_manager):
    """Test rate limiting functionality."""
    # Configure rate limits
    api_manager.config.rate_limit = 2
    api_manager.config.burst_limit = 3
    
    # Test within limits
    for _ in range(2):
        assert api_manager.check_rate_limit() is True
    
    # Test exceeding limit
    assert api_manager.check_rate_limit() is False
    
    # Test rate limit status
    status = api_manager.get_rate_limit_status()
    assert status["remaining"] == 0
    assert "reset_time" in status

@responses.activate
def test_service_integration(mock_api_response):
    """Test integration with different services."""
    services = ["alphafold", "esmfold", "rosettafold"]
    
    for service in services:
        # Mock service endpoint
        responses.add(
            responses.POST,
            f"https://api.example.com/{service}/predict",
            json=mock_api_response,
            status=200
        )
        
        # Test service
        manager = APIManager(service=service, api_key="test_key")
        result = manager.predict_structure("SEQUENCE")
        
        assert result["prediction_id"] == mock_api_response["prediction_id"]
        
        # Test service metrics
        metrics = manager.get_service_metrics()
        assert "status" in metrics
        assert "avg_response_time" in metrics

@patch("proteus.core.batch.BatchProcessor")
def test_batch_processing(mock_batch_processor, api_manager):
    """Test batch processing with API integration."""
    # Configure mock batch processor
    mock_processor = MagicMock()
    mock_batch_processor.return_value = mock_processor
    mock_processor.process_batch.return_value = [mock_api_response]
    
    # Test batch processing
    sequences = ["SEQ1", "SEQ2", "SEQ3"]
    processor = mock_batch_processor(
        api_manager=api_manager,
        max_batch_size=2
    )
    
    results = processor.process_batch(sequences=sequences)
    assert len(results) == 1
    assert results[0]["prediction_id"] == mock_api_response["prediction_id"]

def test_api_caching(api_manager, mock_api_response, temp_dir):
    """Test API response caching."""
    # Configure caching
    cache_dir = temp_dir / "cache"
    api_manager.configure_cache(
        cache_dir=cache_dir,
        max_size=100
    )
    
    # Mock cached response
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    api_manager.cache_response(sequence, mock_api_response)
    
    # Test cache hit
    cached_result = api_manager.get_cached_response(sequence)
    assert cached_result == mock_api_response
    
    # Test cache miss
    assert api_manager.get_cached_response("DIFFERENT") is None

def test_api_monitoring(api_manager):
    """Test API monitoring functionality."""
    # Initialize monitoring
    api_manager.init_monitoring()
    
    # Simulate requests
    for _ in range(3):
        try:
            api_manager.predict_structure("SEQUENCE")
        except:
            pass
    
    # Check metrics
    metrics = api_manager.get_monitoring_metrics()
    assert metrics["total_requests"] == 3
    assert "success_rate" in metrics
    assert "avg_response_time" in metrics

def test_api_validation(api_manager):
    """Test API input validation."""
    # Test sequence validation
    with pytest.raises(ValueError):
        api_manager.validate_sequence("")
    
    with pytest.raises(ValueError):
        api_manager.validate_sequence("123")
    
    # Test valid sequence
    valid_sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    assert api_manager.validate_sequence(valid_sequence) is True
    
    # Test configuration validation
    with pytest.raises(ValueError):
        api_manager.validate_config({"timeout": -1})
    
    with pytest.raises(ValueError):
        api_manager.validate_config({"rate_limit": 0}) 