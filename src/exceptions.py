"""Custom exceptions for the multi-model analysis framework."""

class MultiModelError(Exception):
    """Base exception for all framework errors."""
    pass

class ModelError(MultiModelError):
    """Base exception for model-related errors."""
    pass

class ModelInitializationError(ModelError):
    """Raised when model initialization fails."""
    pass

class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    pass

class ModelVersionError(ModelError):
    """Raised when model versioning operations fail."""
    pass

class ModelConfigError(ModelError):
    """Raised when model configuration is invalid."""
    pass

class ModelResourceError(ModelError):
    """Raised when model resource allocation fails."""
    pass

class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass

class ModelFileError(ModelError):
    """Raised when model file operations fail."""
    pass

class ModelMemoryError(ModelError):
    """Raised when model memory operations fail."""
    pass

class ModelQuantizationError(ModelError):
    """Raised when model quantization fails."""
    pass

class ModelVisualizationError(ModelError):
    """Raised when model visualization fails."""
    pass

class ModelDatabaseError(ModelError):
    """Raised when model database operations fail."""
    pass

class ModelEnsembleError(ModelError):
    """Raised when model ensemble operations fail."""
    pass

class DecoderError(MultiModelError):
    """Raised when model output decoding fails."""
    pass

class TemplateError(MultiModelError):
    """Raised when template-based prediction operations fail."""
    pass

class ConfigurationError(MultiModelError):
    """Raised when configuration is invalid."""
    pass

class ResourceError(MultiModelError):
    """Raised when resource operations fail."""
    pass

class ValidationError(MultiModelError):
    """Raised when input validation fails."""
    pass

class FileError(MultiModelError):
    """Raised when file operations fail."""
    pass

class MemoryError(MultiModelError):
    """Raised when memory operations fail."""
    pass

class QuantizationError(MultiModelError):
    """Raised when model quantization fails."""
    pass

class VisualizationError(MultiModelError):
    """Raised when visualization operations fail."""
    pass

class DatabaseError(MultiModelError):
    """Raised when database operations fail."""
    pass

class EnsembleError(MultiModelError):
    """Raised when ensemble operations fail."""
    pass 