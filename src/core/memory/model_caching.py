"""Model weight caching system."""

import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import logging

from src.exceptions import CacheError

logger = logging.getLogger(__name__)

class CacheEntry:
    """Represents a cached model weight entry."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        weights_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize cache entry.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            weights_path: Path to model weights file
            metadata: Optional metadata dictionary
        """
        self.model_name = model_name
        self.model_type = model_type
        self.weights_path = weights_path
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of weights file."""
        sha256_hash = hashlib.sha256()
        with open(self.weights_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'weights_path': self.weights_path,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create entry from dictionary."""
        entry = cls(
            model_name=data['model_name'],
            model_type=data['model_type'],
            weights_path=data['weights_path'],
            metadata=data['metadata']
        )
        entry.created_at = datetime.fromisoformat(data['created_at'])
        entry.checksum = data['checksum']
        return entry

class ModelCache:
    """Manages caching of model weights."""
    
    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int = 10 * 1024 * 1024 * 1024,  # 10GB
        max_age_days: int = 30
    ):
        """Initialize cache.
        
        Args:
            cache_dir: Directory to store cached weights
            max_size_bytes: Maximum cache size in bytes
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self.max_age_days = max_age_days
        self.entries: Dict[str, CacheEntry] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache entries from disk."""
        cache_file = self.cache_dir / 'cache.json'
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    self.entries = {
                        checksum: CacheEntry.from_dict(entry_data)
                        for checksum, entry_data in data.items()
                    }
                logger.info(f"Loaded {len(self.entries)} cache entries")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.entries = {}
    
    def _save_cache(self):
        """Save cache entries to disk."""
        cache_file = self.cache_dir / 'cache.json'
        try:
            with open(cache_file, 'w') as f:
                json.dump(
                    {checksum: entry.to_dict() for checksum, entry in self.entries.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _validate_weights(self, weights_path: str) -> bool:
        """Validate model weights file.
        
        Args:
            weights_path: Path to weights file
            
        Returns:
            True if weights are valid, False otherwise
        """
        try:
            # Try loading weights
            torch.load(weights_path)
            return True
        except Exception as e:
            logger.error(f"Error validating weights {weights_path}: {e}")
            return False
    
    def _get_cache_size(self) -> int:
        """Get total size of cached weights in bytes."""
        total_size = 0
        for entry in self.entries.values():
            try:
                total_size += os.path.getsize(entry.weights_path)
            except OSError:
                continue
        return total_size
    
    def add(
        self,
        model_name: str,
        model_type: str,
        weights_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheEntry:
        """Add model weights to cache.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            weights_path: Path to model weights file
            metadata: Optional metadata dictionary
            
        Returns:
            CacheEntry object
            
        Raises:
            CacheError: If weights file is invalid or cache is full
        """
        if not os.path.exists(weights_path):
            raise CacheError(f"Weights file not found: {weights_path}")
        
        if not self._validate_weights(weights_path):
            raise CacheError(f"Invalid weights file: {weights_path}")
        
        # Create cache entry
        entry = CacheEntry(model_name, model_type, weights_path, metadata)
        
        # Check cache size
        if self._get_cache_size() + os.path.getsize(weights_path) > self.max_size_bytes:
            raise CacheError("Cache size limit exceeded")
        
        # Add entry
        self.entries[entry.checksum] = entry
        self._save_cache()
        
        logger.info(f"Added {model_name} to cache")
        return entry
    
    def get(self, checksum: str) -> torch.Tensor:
        """Get model weights from cache.
        
        Args:
            checksum: Checksum of cached weights
            
        Returns:
            Model weights tensor
            
        Raises:
            CacheError: If weights not found or invalid
        """
        if checksum not in self.entries:
            raise CacheError(f"Cache entry not found: {checksum}")
        
        entry = self.entries[checksum]
        if not os.path.exists(entry.weights_path):
            raise CacheError(f"Weights file not found: {entry.weights_path}")
        
        if not self._validate_weights(entry.weights_path):
            raise CacheError(f"Invalid weights file: {entry.weights_path}")
        
        try:
            weights = torch.load(entry.weights_path)
            logger.info(f"Retrieved {entry.model_name} from cache")
            return weights
        except Exception as e:
            raise CacheError(f"Error loading weights: {e}")
    
    def cleanup(self, max_age_days: Optional[int] = None):
        """Clean up old cache entries.
        
        Args:
            max_age_days: Maximum age of entries in days
        """
        if max_age_days is None:
            max_age_days = self.max_age_days
        
        current_time = datetime.now()
        entries_to_remove = []
        
        for checksum, entry in self.entries.items():
            age = (current_time - entry.created_at).days
            if age > max_age_days:
                entries_to_remove.append(checksum)
                try:
                    os.remove(entry.weights_path)
                except OSError:
                    pass
        
        for checksum in entries_to_remove:
            del self.entries[checksum]
        
        if entries_to_remove:
            self._save_cache()
            logger.info(f"Removed {len(entries_to_remove)} old cache entries")
    
    def update_metadata(self, checksum: str, metadata: Dict[str, Any]):
        """Update metadata for cache entry.
        
        Args:
            checksum: Checksum of cache entry
            metadata: New metadata dictionary
            
        Raises:
            CacheError: If entry not found
        """
        if checksum not in self.entries:
            raise CacheError(f"Cache entry not found: {checksum}")
        
        entry = self.entries[checksum]
        entry.metadata.update(metadata)
        self._save_cache()
    
    def list_entries(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[CacheEntry]:
        """List cache entries with optional filtering.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            
        Returns:
            List of matching cache entries
        """
        entries = self.entries.values()
        
        if model_name:
            entries = [e for e in entries if e.model_name == model_name]
        
        if model_type:
            entries = [e for e in entries if e.model_type == model_type]
        
        return list(entries) 