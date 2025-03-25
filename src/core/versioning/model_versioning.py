"""Model versioning system for managing model versions and tracking changes."""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from src.exceptions import ModelVersionError

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    version_id: str
    model_name: str
    model_type: str
    model_path: str
    config_path: Optional[str]
    created_at: str
    description: str
    metadata: Dict[str, Any]
    checksum: str
    parent_version: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create version from dictionary."""
        return cls(**data)

class ModelVersionManager:
    """Manages model versions and their metadata."""
    
    def __init__(self, version_dir: str):
        """
        Initialize version manager.
        
        Args:
            version_dir: Directory to store version information
        """
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.version_dir / 'versions.json'
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load version information from file."""
        try:
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    versions_data = json.load(f)
                    self.versions = {
                        v['version_id']: ModelVersion.from_dict(v)
                        for v in versions_data
                    }
            else:
                self.versions = {}
        except Exception as e:
            logger.error(f"Failed to load versions: {str(e)}")
            raise ModelVersionError(f"Failed to load versions: {str(e)}")
    
    def _save_versions(self) -> None:
        """Save version information to file."""
        try:
            versions_data = [v.to_dict() for v in self.versions.values()]
            with open(self.versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {str(e)}")
            raise ModelVersionError(f"Failed to save versions: {str(e)}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 checksum
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def add_version(
        self,
        model_name: str,
        model_type: str,
        model_path: str,
        config_path: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a new model version.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            model_path: Path to model weights
            config_path: Optional path to config file
            description: Version description
            metadata: Additional metadata
            parent_version: ID of parent version
            tags: List of tags
            
        Returns:
            Version ID
        """
        try:
            # Validate paths
            if not os.path.exists(model_path):
                raise ModelVersionError(f"Model file not found: {model_path}")
            if config_path and not os.path.exists(config_path):
                raise ModelVersionError(f"Config file not found: {config_path}")
            
            # Generate version ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            version_id = f"{model_name}_{timestamp}"
            
            # Calculate checksum
            checksum = self._calculate_checksum(model_path)
            
            # Create version object
            version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                model_type=model_type,
                model_path=model_path,
                config_path=config_path,
                created_at=datetime.utcnow().isoformat(),
                description=description,
                metadata=metadata or {},
                checksum=checksum,
                parent_version=parent_version,
                tags=tags or []
            )
            
            # Store version
            self.versions[version_id] = version
            self._save_versions()
            
            logger.info(f"Added new version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to add version: {str(e)}")
            raise ModelVersionError(f"Failed to add version: {str(e)}")
    
    def get_version(self, version_id: str) -> ModelVersion:
        """
        Get version information.
        
        Args:
            version_id: Version ID
            
        Returns:
            ModelVersion object
        """
        if version_id not in self.versions:
            raise ModelVersionError(f"Version not found: {version_id}")
        return self.versions[version_id]
    
    def list_versions(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        List versions with optional filtering.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            tag: Filter by tag
            
        Returns:
            List of matching versions
        """
        versions = self.versions.values()
        
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        if tag:
            versions = [v for v in versions if tag in v.tags]
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def delete_version(self, version_id: str) -> None:
        """
        Delete a version.
        
        Args:
            version_id: Version ID
        """
        try:
            if version_id not in self.versions:
                raise ModelVersionError(f"Version not found: {version_id}")
            
            # Check if version is parent of other versions
            for version in self.versions.values():
                if version.parent_version == version_id:
                    raise ModelVersionError(
                        f"Cannot delete version {version_id} as it is a parent of version {version.version_id}"
                    )
            
            del self.versions[version_id]
            self._save_versions()
            
            logger.info(f"Deleted version: {version_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete version: {str(e)}")
            raise ModelVersionError(f"Failed to delete version: {str(e)}")
    
    def add_tag(self, version_id: str, tag: str) -> None:
        """
        Add a tag to a version.
        
        Args:
            version_id: Version ID
            tag: Tag to add
        """
        try:
            version = self.get_version(version_id)
            if tag not in version.tags:
                version.tags.append(tag)
                self._save_versions()
                logger.info(f"Added tag '{tag}' to version {version_id}")
        except Exception as e:
            logger.error(f"Failed to add tag: {str(e)}")
            raise ModelVersionError(f"Failed to add tag: {str(e)}")
    
    def remove_tag(self, version_id: str, tag: str) -> None:
        """
        Remove a tag from a version.
        
        Args:
            version_id: Version ID
            tag: Tag to remove
        """
        try:
            version = self.get_version(version_id)
            if tag in version.tags:
                version.tags.remove(tag)
                self._save_versions()
                logger.info(f"Removed tag '{tag}' from version {version_id}")
        except Exception as e:
            logger.error(f"Failed to remove tag: {str(e)}")
            raise ModelVersionError(f"Failed to remove tag: {str(e)}")
    
    def get_version_history(self, version_id: str) -> List[ModelVersion]:
        """
        Get version history (ancestors).
        
        Args:
            version_id: Version ID
            
        Returns:
            List of versions in order from oldest to newest
        """
        history = []
        current_version = self.get_version(version_id)
        
        while current_version.parent_version:
            history.append(current_version)
            current_version = self.get_version(current_version.parent_version)
        
        history.append(current_version)
        return list(reversed(history))
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            v1 = self.get_version(version_id1)
            v2 = self.get_version(version_id2)
            
            comparison = {
                'version1': v1.to_dict(),
                'version2': v2.to_dict(),
                'differences': {
                    'checksum': v1.checksum != v2.checksum,
                    'metadata': {
                        k: v1.metadata.get(k) != v2.metadata.get(k)
                        for k in set(v1.metadata.keys()) | set(v2.metadata.keys())
                    },
                    'tags': set(v1.tags) != set(v2.tags)
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {str(e)}")
            raise ModelVersionError(f"Failed to compare versions: {str(e)}")