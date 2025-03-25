"""Test suite for the Proteus framework."""

from .structure import *
from .models import *
from .core import *
from .performance import *

__all__ = [
    # Structure tests
    'TestStructureComparison',
    'TestAdvancedVisualization',
    'TestVisualization',
    'TestPlotting',
    
    # Model tests
    'TestModelInterface',
    'TestColabfoldInterface',
    'TestModelVersioning',
    'TestModelCaching',
    'TestModelVersioningAndCaching',
    'TestAPIIntegration',
    'TestEnsemble',
    'TestEnsembleAndVersioning',
    
    # Core tests
    'TestCore',
    'TestFileProcessing',
    'TestLoggingConfig',
    'TestDocAutomation',
    'TestPDBEncoder',
    'TestQuantization',
    'TestDatabaseManager',
    
    # Performance tests
    'TestBatchProcessing',
    'TestBatchProcessor',
    'TestMemoryManagement',
    'TestDistributedInference',
    'TestBenchmarking',
    
    # Benchmarking utilities
    'ModelMetrics',
    'StructureMetrics',
    'PerformanceMetrics',
    'BenchmarkDataset',
    'ProteinDataset',
    'PerformanceTracker'
]