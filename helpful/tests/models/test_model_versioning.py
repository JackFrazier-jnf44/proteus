"""Unit tests for model versioning system."""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from multi_model_analysis.utils.model_versioning import (
    ModelVersion,
    ModelVersionManager
)
from multi_model_analysis.exceptions import ModelVersionError

class TestModelVersioning(unittest.TestCase):
    """Test cases for model versioning system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_dir = Path(self.temp_dir)
        
        # Create test model file
        self.model_file = self.version_dir / 'test_model.pt'
        self.model_file.write_text('Test model weights')
        
        # Create test config file
        self.config_file = self.version_dir / 'test_config.yaml'
        self.config_file.write_text('Test config')
        
        # Initialize version manager
        self.version_manager = ModelVersionManager(str(self.version_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_version(self):
        """Test adding a new version."""
        version_id = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file),
            config_path=str(self.config_file),
            description='Test version',
            metadata={'test_key': 'test_value'},
            tags=['test_tag']
        )
        
        # Check if version was added
        self.assertIn(version_id, self.version_manager.versions)
        version = self.version_manager.versions[version_id]
        
        # Check version attributes
        self.assertEqual(version.model_name, 'test_model')
        self.assertEqual(version.model_type, 'test_type')
        self.assertEqual(version.description, 'Test version')
        self.assertEqual(version.metadata['test_key'], 'test_value')
        self.assertEqual(version.tags, ['test_tag'])
        
        # Check if checksum was calculated
        self.assertIsNotNone(version.checksum)
    
    def test_add_version_invalid_path(self):
        """Test adding version with invalid path."""
        with self.assertRaises(ModelVersionError):
            self.version_manager.add_version(
                model_name='test_model',
                model_type='test_type',
                model_path='nonexistent.pt'
            )
    
    def test_get_version(self):
        """Test getting version information."""
        version_id = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file)
        )
        
        version = self.version_manager.get_version(version_id)
        self.assertEqual(version.version_id, version_id)
        
        # Test getting non-existent version
        with self.assertRaises(ModelVersionError):
            self.version_manager.get_version('nonexistent')
    
    def test_list_versions(self):
        """Test listing versions with filters."""
        # Add multiple versions
        version_ids = []
        for i in range(3):
            version_id = self.version_manager.add_version(
                model_name=f'test_model_{i}',
                model_type='test_type',
                model_path=str(self.model_file),
                tags=['test_tag']
            )
            version_ids.append(version_id)
        
        # Test listing all versions
        all_versions = self.version_manager.list_versions()
        self.assertEqual(len(all_versions), 3)
        
        # Test filtering by model name
        filtered_versions = self.version_manager.list_versions(model_name='test_model_0')
        self.assertEqual(len(filtered_versions), 1)
        
        # Test filtering by model type
        filtered_versions = self.version_manager.list_versions(model_type='test_type')
        self.assertEqual(len(filtered_versions), 3)
        
        # Test filtering by tag
        filtered_versions = self.version_manager.list_versions(tag='test_tag')
        self.assertEqual(len(filtered_versions), 3)
    
    def test_delete_version(self):
        """Test deleting a version."""
        version_id = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file)
        )
        
        # Test deleting version
        self.version_manager.delete_version(version_id)
        self.assertNotIn(version_id, self.version_manager.versions)
        
        # Test deleting non-existent version
        with self.assertRaises(ModelVersionError):
            self.version_manager.delete_version('nonexistent')
    
    def test_delete_version_with_children(self):
        """Test deleting a version that has child versions."""
        # Create parent version
        parent_id = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file)
        )
        
        # Create child version
        child_id = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file),
            parent_version=parent_id
        )
        
        # Test deleting parent version
        with self.assertRaises(ModelVersionError):
            self.version_manager.delete_version(parent_id)
    
    def test_tag_management(self):
        """Test adding and removing tags."""
        version_id = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file)
        )
        
        # Test adding tag
        self.version_manager.add_tag(version_id, 'test_tag')
        version = self.version_manager.get_version(version_id)
        self.assertIn('test_tag', version.tags)
        
        # Test removing tag
        self.version_manager.remove_tag(version_id, 'test_tag')
        version = self.version_manager.get_version(version_id)
        self.assertNotIn('test_tag', version.tags)
    
    def test_version_history(self):
        """Test getting version history."""
        # Create version chain
        version_ids = []
        parent_id = None
        for i in range(3):
            version_id = self.version_manager.add_version(
                model_name='test_model',
                model_type='test_type',
                model_path=str(self.model_file),
                parent_version=parent_id
            )
            version_ids.append(version_id)
            parent_id = version_id
        
        # Test getting history
        history = self.version_manager.get_version_history(version_ids[-1])
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].version_id, version_ids[0])
        self.assertEqual(history[-1].version_id, version_ids[-1])
    
    def test_compare_versions(self):
        """Test comparing versions."""
        # Create two versions
        version_id1 = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file),
            metadata={'key1': 'value1'}
        )
        
        version_id2 = self.version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            model_path=str(self.model_file),
            metadata={'key1': 'value2'}
        )
        
        # Compare versions
        comparison = self.version_manager.compare_versions(version_id1, version_id2)
        
        # Check comparison results
        self.assertEqual(comparison['version1']['version_id'], version_id1)
        self.assertEqual(comparison['version2']['version_id'], version_id2)
        self.assertTrue(comparison['differences']['metadata']['key1'])
        self.assertFalse(comparison['differences']['checksum'])

if __name__ == '__main__':
    unittest.main() 