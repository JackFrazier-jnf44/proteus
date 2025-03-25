"""Configuration validation utilities."""

import torch
import psutil
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.exceptions import ValidationError

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration settings and requirements."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize config validator.
        
        Args:
            config: Configuration dictionary to validate
        """
        self.config = config
        self.required_packages = {
            'torch': '1.8.0',
            'numpy': '1.19.0',
            'biopython': '1.79',
            'scipy': '1.6.0'
        }
        
    def validate_dependencies(self):
        """Validate package dependencies and versions."""
        try:
            import pkg_resources
            
            for package, min_version in self.required_packages.items():
                try:
                    installed = pkg_resources.get_distribution(package)
                    if pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(min_version):
                        raise ValidationError(
                            f"Package {package} version {installed.version} is below "
                            f"minimum required version {min_version}"
                        )
                except pkg_resources.DistributionNotFound:
                    raise ValidationError(f"Required package {package} is not installed")
                    
            logger.info("All package dependencies validated successfully")
            
        except Exception as e:
            logger.error(f"Dependency validation failed: {str(e)}")
            raise ValidationError(f"Dependency validation failed: {str(e)}")
        
    def validate_hardware_requirements(self):
        """Validate hardware requirements."""
        try:
            # Check CPU requirements
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count < self.config.get('min_cpu_cores', 1):
                raise ValidationError(
                    f"Insufficient CPU cores. Found {cpu_count}, "
                    f"required {self.config['min_cpu_cores']}"
                )
            
            # Check memory requirements
            memory = psutil.virtual_memory()
            min_memory_gb = self.config.get('min_memory_gb', 8)
            available_memory_gb = memory.total / (1024 ** 3)
            if available_memory_gb < min_memory_gb:
                raise ValidationError(
                    f"Insufficient system memory. Found {available_memory_gb:.1f}GB, "
                    f"required {min_memory_gb}GB"
                )
            
            # Check GPU requirements if needed
            if self.config.get('use_gpu', False):
                if not torch.cuda.is_available():
                    raise ValidationError("GPU required but CUDA is not available")
                    
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory.append(gpu_props.total_memory / (1024 ** 3))
                    
                min_gpu_memory_gb = self.config.get('min_gpu_memory_gb', 8)
                if not any(mem >= min_gpu_memory_gb for mem in gpu_memory):
                    raise ValidationError(
                        f"No GPU found with sufficient memory. Required {min_gpu_memory_gb}GB, "
                        f"found GPUs with memory: {[f'{mem:.1f}GB' for mem in gpu_memory]}"
                    )
            
            # Check disk space requirements
            if 'output_dir' in self.config:
                output_dir = Path(self.config['output_dir'])
                disk = psutil.disk_usage(str(output_dir))
                min_disk_space_gb = self.config.get('min_disk_space_gb', 10)
                available_space_gb = disk.free / (1024 ** 3)
                if available_space_gb < min_disk_space_gb:
                    raise ValidationError(
                        f"Insufficient disk space in output directory. "
                        f"Found {available_space_gb:.1f}GB, required {min_disk_space_gb}GB"
                    )
            
            logger.info("All hardware requirements validated successfully")
            
        except Exception as e:
            logger.error(f"Hardware validation failed: {str(e)}")
            raise ValidationError(f"Hardware validation failed: {str(e)}") 