"""
Example script demonstrating API integration capabilities of the Proteus framework.
"""

import os
from pathlib import Path
from src.interfaces import (
    APIManager,
    APIConfig,
    ModelInterface,
    BaseModelConfig
)
from src.core.batch import BatchProcessor
from src.core.logging import setup_logging

def basic_api_integration():
    """Demonstrate basic API integration functionality."""
    # Initialize API manager
    api_manager = APIManager(
        service="alphafold",
        api_key=os.getenv("ALPHAFOLD_API_KEY")
    )
    
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    try:
        # Make prediction request
        result = api_manager.predict_structure(sequence)
        
        # Print results
        print("\nBasic API integration results:")
        print(f"Prediction ID: {result['prediction_id']}")
        print(f"Confidence score: {result['confidence']:.3f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"\nAPI request failed: {e}")

def advanced_api_configuration():
    """Demonstrate advanced API configuration."""
    # Configure API settings
    config = APIConfig(
        service="alphafold",
        api_key=os.getenv("ALPHAFOLD_API_KEY"),
        base_url="https://api.example.com",
        max_retries=3,
        timeout=300,
        rate_limit=60,
        burst_limit=120,
        cooldown=60,
        backoff_factor=1.5
    )
    
    # Initialize with configuration
    api_manager = APIManager(config)
    
    # Test sequences
    sequences = [
        "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    try:
        # Process sequences with monitoring
        for i, sequence in enumerate(sequences, 1):
            print(f"\nProcessing sequence {i}...")
            
            result = api_manager.predict_structure_with_retry(
                sequence=sequence,
                backoff_factor=1.5
            )
            
            print(f"Prediction ID: {result['prediction_id']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Retries: {result['retry_count']}")
            
            # Check rate limit status
            status = api_manager.get_rate_limit_status()
            print(f"Remaining requests: {status['remaining']}")
            print(f"Reset time: {status['reset_time']}")
            
    except Exception as e:
        print(f"\nAPI processing failed: {e}")

def api_service_integration():
    """Demonstrate integration with different API services."""
    # Initialize managers for different services
    services = {
        "alphafold": APIManager(
            service="alphafold",
            api_key=os.getenv("ALPHAFOLD_API_KEY")
        ),
        "esmfold": APIManager(
            service="esmfold",
            api_key=os.getenv("ESMFOLD_API_KEY")
        ),
        "rosettafold": APIManager(
            service="rosettafold",
            api_key=os.getenv("ROSETTAFOLD_API_KEY")
        )
    }
    
    # Test sequence
    sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
    
    print("\nComparing predictions across services:")
    
    for service_name, manager in services.items():
        try:
            print(f"\n{service_name.upper()} prediction:")
            result = manager.predict_structure(sequence)
            
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
            # Get service-specific metrics
            metrics = manager.get_service_metrics()
            print(f"Service status: {metrics['status']}")
            print(f"Average response time: {metrics['avg_response_time']:.2f}s")
            
        except Exception as e:
            print(f"Failed to get prediction: {e}")

def batch_api_processing():
    """Demonstrate batch processing with API integration."""
    # Initialize API manager
    api_manager = APIManager(
        service="alphafold",
        api_key=os.getenv("ALPHAFOLD_API_KEY")
    )
    
    # Configure batch processor
    processor = BatchProcessor(
        api_manager=api_manager,
        max_batch_size=10
    )
    
    # Generate test sequences
    sequences = [
        "".join(["ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(length)])
        for length in [50, 75, 100]
    ]
    
    try:
        # Process batch with progress tracking
        def progress_callback(batch_id, status, progress):
            print(f"Batch {batch_id}: {status} ({progress:.1f}%)")
        
        results = processor.process_batch(
            sequences=sequences,
            callbacks=[progress_callback]
        )
        
        print("\nBatch processing results:")
        for i, result in enumerate(results, 1):
            print(f"\nSequence {i}:")
            print(f"Prediction ID: {result['prediction_id']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processing time: {result['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"\nBatch processing failed: {e}")

def api_error_handling():
    """Demonstrate API error handling and recovery."""
    # Initialize manager with retry configuration
    api_manager = APIManager(
        service="alphafold",
        api_key=os.getenv("ALPHAFOLD_API_KEY"),
        max_retries=3,
        retry_strategy="exponential"
    )
    
    # Test scenarios
    scenarios = [
        ("MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", "Valid sequence"),
        ("INVALID123", "Invalid sequence"),
        ("X" * 1000, "Sequence too long")
    ]
    
    print("\nTesting error handling:")
    
    for sequence, description in scenarios:
        print(f"\nScenario: {description}")
        
        try:
            result = api_manager.predict_structure_with_retry(
                sequence=sequence,
                backoff_factor=1.5
            )
            
            print("Prediction successful:")
            print(f"ID: {result['prediction_id']}")
            print(f"Retries: {result['retry_count']}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Recovery strategy: ", end="")
            
            try:
                # Attempt recovery based on error type
                if "invalid sequence" in str(e).lower():
                    print("Skipping invalid sequence")
                elif "too long" in str(e).lower():
                    print("Splitting sequence into chunks")
                else:
                    print("Using fallback service")
            
            except Exception as recovery_error:
                print(f"Recovery failed: {recovery_error}")

def main():
    """Run API integration examples."""
    # Setup logging
    setup_logging(level="INFO")
    
    print("Running API integration examples...")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    print("\n1. Basic API Integration")
    basic_api_integration()
    
    print("\n2. Advanced API Configuration")
    advanced_api_configuration()
    
    print("\n3. API Service Integration")
    api_service_integration()
    
    print("\n4. Batch API Processing")
    batch_api_processing()
    
    print("\n5. API Error Handling")
    api_error_handling()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 