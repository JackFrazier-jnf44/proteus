"""Unit tests for model weight caching system."""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
import torch
import hashlib

from multi_model_analysis.utils.model_caching import (
    ModelCache,
    CacheEntry,
    CacheError
)

class TestModelCaching(unittest.TestCase):
    """Test cases for model weight caching system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / 'cache'
        self.cache_dir.mkdir()
        
        # Create test model weights
        self.model_weights = torch.randn(10, 10)
        self.model_path = self.cache_dir / 'test_model.pt'
        torch.save(self.model_weights, self.model_path)
        
        # Initialize cache
        self.cache = ModelCache(str(self.cache_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            model_name='test_model',
            model_type='test_type',
            weights_path=str(self.model_path),
            metadata={'test_key': 'test_value'}
        )
        
        # Check entry attributes
        self.assertEqual(entry.model_name, 'test_model')
        self.assertEqual(entry.model_type, 'test_type')
        self.assertEqual(entry.weights_path, str(self.model_path))
        self.assertEqual(entry.metadata['test_key'], 'test_value')
        self.assertIsNotNone(entry.checksum)
        self.assertIsNotNone(entry.created_at)
    
    def test_add_to_cache(self):
        """Test adding model weights to cache."""
        entry = self.cache.add(
            model_name='test_model',
            model_type='test_type',
            weights_path=str(self.model_path),
            metadata={'test_key': 'test_value'}
        )
        
        # Check if entry was added
        self.assertIn(entry.checksum, self.cache.entries)
        cached_entry = self.cache.entries[entry.checksum]
        
        # Check entry attributes
        self.assertEqual(cached_entry.model_name, 'test_model')
        self.assertEqual(cached_entry.model_type, 'test_type')
        self.assertEqual(cached_entry.metadata['test_key'], 'test_value')
    
    def test_add_to_cache_invalid_path(self):
        """Test adding invalid weights path to cache."""
        with self.assertRaises(CacheError):
            self.cache.add(
                model_name='test_model',
                model_type='test_type',
                weights_path='nonexistent.pt'
            )
    
    def test_get_from_cache(self):
        """Test retrieving model weights from cache."""
        # Add model to cache
        entry = self.cache.add(
            model_name='test_model',
            model_type='test_type',
            weights_path=str(self.model_path)
        )
        
        # Retrieve weights
        cached_weights = self.cache.get(entry.checksum)
        
        # Check if weights match
        self.assertTrue(torch.allclose(cached_weights, self.model_weights))
        
        # Test getting non-existent weights
        with self.assertRaises(CacheError):
            self.cache.get('nonexistent_checksum')
    
    def test_cache_cleanup(self):
        """Test cleaning up cache entries."""
        # Add multiple entries
        entries = []
        for i in range(3):
            entry = self.cache.add(
                model_name=f'test_model_{i}',
                model_type='test_type',
                weights_path=str(self.model_path)
            )
            entries.append(entry)
        
        # Clean up old entries
        self.cache.cleanup(max_age_days=0)
        
        # Check if entries were removed
        self.assertEqual(len(self.cache.entries), 0)
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        # Set small size limit
        self.cache.max_size_bytes = 1000
        
        # Add large model
        with self.assertRaises(CacheError):
            self.cache.add(
                model_name='test_model',
                model_type='test_type',
                weights_path=str(self.model_path)
            )
    
    def test_cache_metadata(self):
        """Test cache metadata management."""
        # Add model with metadata
        entry = self.cache.add(
            model_name='test_model',
            model_type='test_type',
            weights_path=str(self.model_path),
            metadata={'test_key': 'test_value'}
        )
        
        # Update metadata
        self.cache.update_metadata(entry.checksum, {'new_key': 'new_value'})
        
        # Check updated metadata
        updated_entry = self.cache.entries[entry.checksum]
        self.assertEqual(updated_entry.metadata['new_key'], 'new_value')
        self.assertEqual(updated_entry.metadata['test_key'], 'test_value')
    
    def test_cache_validation(self):
        """Test cache entry validation."""
        # Add model to cache
        entry = self.cache.add(
            model_name='test_model',
            model_type='test_type',
            weights_path=str(self.model_path)
        )
        
        # Corrupt weights file
        with open(self.model_path, 'wb') as f:
            f.write(b'corrupted')
        
        # Test validation
        with self.assertRaises(CacheError):
            self.cache.get(entry.checksum)
    
    def test_cache_persistence(self):
        """Test cache persistence between sessions."""
        # Add model to cache
        entry = self.cache.add(
            model_name='test_model',
            model_type='test_type',
            weights_path=str(self.model_path)
        )
        
        # Create new cache instance
        new_cache = ModelCache(str(self.cache_dir))
        
        # Check if entry persists
        self.assertIn(entry.checksum, new_cache.entries)
        cached_entry = new_cache.entries[entry.checksum]
        self.assertEqual(cached_entry.model_name, 'test_model')
        self.assertEqual(cached_entry.model_type, 'test_type')

if __name__ == '__main__':
    unittest.main() 