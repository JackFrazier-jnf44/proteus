"""Performance tracking and analysis utilities."""

import time
import json
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
from .metrics import ModelMetrics, StructureMetrics, PerformanceMetrics

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    model_name: str
    dataset_name: str
    timestamp: float
    model_metrics: ModelMetrics
    structure_metrics: StructureMetrics
    performance_metrics: PerformanceMetrics
    metadata: Dict[str, any] = field(default_factory=dict)

@dataclass
class PerformanceTracker:
    """Tracks and analyzes performance metrics during benchmarking."""
    
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: Optional[float] = None
    current_memory: float = 0.0
    peak_memory: float = 0.0
    gpu_peak_memory: float = 0.0
    
    def start_tracking(self) -> None:
        """Start tracking performance metrics."""
        self.start_time = time.time()
        self.current_memory = 0.0
        self.peak_memory = 0.0
        self.gpu_peak_memory = 0.0
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    
    def update_memory_stats(self) -> None:
        """Update memory usage statistics."""
        process = psutil.Process()
        self.current_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
        self.peak_memory = max(self.peak_memory, self.current_memory)
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # Convert to GB
            self.gpu_peak_memory = max(self.gpu_peak_memory, current_gpu)
    
    def get_performance_data(self) -> Dict[str, float]:
        """Get current performance metrics."""
        self.update_memory_stats()
        
        data = {
            "inference_time": time.time() - self.start_time,
            "peak_memory_usage": self.peak_memory,
            "gpu_memory_usage": self.gpu_peak_memory,
            "cpu_utilization": psutil.cpu_percent(),
        }
        
        if torch.cuda.is_available():
            data["gpu_utilization"] = torch.cuda.utilization()
        
        return data
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the tracker."""
        self.results.append(result)
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked results."""
        if not self.results:
            return {}
        
        stats = {}
        
        # Group results by model
        for model_name in set(r.model_name for r in self.results):
            model_results = [r for r in self.results if r.model_name == model_name]
            
            model_stats = {
                "avg_rmsd": np.mean([r.model_metrics.rmsd for r in model_results]),
                "avg_tm_score": np.mean([r.model_metrics.tm_score for r in model_results]),
                "avg_gdt_ts": np.mean([r.model_metrics.gdt_ts for r in model_results]),
                "avg_processing_time": np.mean([r.model_metrics.processing_time for r in model_results]),
                "avg_memory_usage": np.mean([r.model_metrics.memory_usage for r in model_results]),
                "avg_throughput": np.mean([r.performance_metrics.throughput for r in model_results]),
            }
            
            stats[model_name] = model_stats
        
        return stats
    
    def save_results(self, path: Union[str, Path]) -> None:
        """Save benchmark results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serialized_results = []
        for result in self.results:
            serialized = {
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "timestamp": result.timestamp,
                "model_metrics": vars(result.model_metrics),
                "structure_metrics": vars(result.structure_metrics),
                "performance_metrics": vars(result.performance_metrics),
                "metadata": result.metadata
            }
            serialized_results.append(serialized)
        
        # Save to JSON
        with open(path, 'w') as f:
            json.dump({
                "results": serialized_results,
                "summary_stats": self.get_summary_stats()
            }, f, indent=2)
    
    @classmethod
    def load_results(cls, path: Union[str, Path]) -> 'PerformanceTracker':
        """Load benchmark results from file."""
        path = Path(path)
        
        with open(path) as f:
            data = json.load(f)
        
        tracker = cls()
        
        # Reconstruct results
        for result_data in data["results"]:
            result = BenchmarkResult(
                model_name=result_data["model_name"],
                dataset_name=result_data["dataset_name"],
                timestamp=result_data["timestamp"],
                model_metrics=ModelMetrics(**result_data["model_metrics"]),
                structure_metrics=StructureMetrics(**result_data["structure_metrics"]),
                performance_metrics=PerformanceMetrics(**result_data["performance_metrics"]),
                metadata=result_data["metadata"]
            )
            tracker.add_result(result)
        
        return tracker 