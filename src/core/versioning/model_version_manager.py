"""Model versioning and checkpointing manager."""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import shutil
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict
import torch
import numpy as np

from src.exceptions import ModelError, ModelVersionError

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints."""
    model_name: str
    version: str
    timestamp: str
    model_hash: str
    training_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    config: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, str]] = None

class ModelVersionManager:
    """Manages model versions and checkpoints."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        model_name: str,
        max_checkpoints: int = 5
    ):
        """Initialize version manager.
        
        Args:
            base_dir: Base directory for checkpoints
            model_name: Name of the model
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = self.base_dir / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load version history
        self.version_history = self._load_version_history()
        
        logger.info(f"Initialized version manager for {model_name} at {self.checkpoint_dir}")
    
    def _load_version_history(self) -> Dict[str, CheckpointMetadata]:
        """Load version history from disk."""
        history_file = self.checkpoint_dir / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                return {
                    version: CheckpointMetadata(**metadata)
                    for version, metadata in history.items()
                }
            except Exception as e:
                logger.error(f"Failed to load version history: {str(e)}")
                raise ModelVersionError(f"Version history loading failed: {str(e)}")
        return {}
    
    def _save_version_history(self) -> None:
        """Save version history to disk."""
        history_file = self.checkpoint_dir / "version_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(
                    {version: asdict(metadata)
                     for version, metadata in self.version_history.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save version history: {str(e)}")
            raise ModelVersionError(f"Version history saving failed: {str(e)}")
    
    def _calculate_model_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Calculate hash of model state."""
        try:
            # Convert state dict to bytes
            state_bytes = b""
            for key in sorted(state_dict.keys()):
                state_bytes += key.encode()
                state_bytes += state_dict[key].cpu().numpy().tobytes()
            
            # Calculate SHA-256 hash
            return hashlib.sha256(state_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate model hash: {str(e)}")
            raise ModelVersionError(f"Model hash calculation failed: {str(e)}")
    
    def create_checkpoint(
        self,
        model: torch.nn.Module,
        version: str,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a new checkpoint.
        
        Args:
            model: Model to checkpoint
            version: Version string
            training_metrics: Optional training metrics
            validation_metrics: Optional validation metrics
            config: Optional model configuration
            dependencies: Optional dependency versions
            
        Returns:
            Checkpoint path
        """
        try:
            # Create checkpoint directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = self.checkpoint_dir / f"{version}_{timestamp}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoint_dir / "model.pt")
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(state_dict)
            
            # Create metadata
            metadata = CheckpointMetadata(
                model_name=self.model_name,
                version=version,
                timestamp=timestamp,
                model_hash=model_hash,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                config=config,
                dependencies=dependencies
            )
            
            # Save metadata
            with open(checkpoint_dir / "metadata.json", 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Update version history
            self.version_history[version] = metadata
            self._save_version_history()
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Created checkpoint for version {version} at {checkpoint_dir}")
            return str(checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {str(e)}")
            raise ModelVersionError(f"Checkpoint creation failed: {str(e)}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        version: str,
        device: Optional[str] = None
    ) -> CheckpointMetadata:
        """Load a checkpoint.
        
        Args:
            model: Model to load checkpoint into
            version: Version to load
            device: Optional device to load model to
            
        Returns:
            Checkpoint metadata
        """
        try:
            if version not in self.version_history:
                raise ModelVersionError(f"Version {version} not found")
            
            metadata = self.version_history[version]
            checkpoint_dir = self.checkpoint_dir / f"{version}_{metadata.timestamp}"
            
            if not checkpoint_dir.exists():
                raise ModelVersionError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            # Load model state
            state_dict = torch.load(checkpoint_dir / "model.pt")
            model.load_state_dict(state_dict)
            
            if device:
                model.to(device)
            
            logger.info(f"Loaded checkpoint for version {version} from {checkpoint_dir}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise ModelVersionError(f"Checkpoint loading failed: {str(e)}")
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest version string."""
        if not self.version_history:
            return None
        return max(self.version_history.keys())
    
    def get_version_metadata(self, version: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific version."""
        return self.version_history.get(version)
    
    def list_versions(self) -> List[str]:
        """List all available versions."""
        return sorted(self.version_history.keys())
    
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        try:
            # Get all checkpoint directories
            checkpoints = sorted(
                self.checkpoint_dir.glob(f"{self.model_name}_*"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Remove excess checkpoints
            for checkpoint in checkpoints[self.max_checkpoints:]:
                shutil.rmtree(checkpoint)
                logger.debug(f"Removed old checkpoint: {checkpoint}")
                
        except Exception as e:
            logger.error(f"Failed to clean up old checkpoints: {str(e)}")
            raise ModelVersionError(f"Checkpoint cleanup failed: {str(e)}")
    
    def verify_checkpoint(self, version: str) -> bool:
        """Verify checkpoint integrity.
        
        Args:
            version: Version to verify
            
        Returns:
            True if checkpoint is valid
        """
        try:
            if version not in self.version_history:
                return False
            
            metadata = self.version_history[version]
            checkpoint_dir = self.checkpoint_dir / f"{version}_{metadata.timestamp}"
            
            if not checkpoint_dir.exists():
                return False
            
            # Verify model file exists
            model_path = checkpoint_dir / "model.pt"
            if not model_path.exists():
                return False
            
            # Verify metadata file exists
            metadata_path = checkpoint_dir / "metadata.json"
            if not metadata_path.exists():
                return False
            
            # Verify model hash
            state_dict = torch.load(model_path)
            current_hash = self._calculate_model_hash(state_dict)
            if current_hash != metadata.model_hash:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify checkpoint: {str(e)}")
            return False 