"""Comprehensive test suite for structural comparison using proteus."""

import os
import pytest
import torch
from pathlib import Path
from datetime import datetime

from src.models import (
    ModelInterface,
    OpenFoldInterface,
    ESMInterface,
    AlphaFoldInterface,
    RoseTTAFoldInterface
)
from src.utils import (
    ModelCache,
    CacheEntry,
    CacheError,
    ModelVersion,
    ModelVersionManager,
    DatabaseManager,
    DatabaseEntry,
    PDBEncoder,
    PDBDecoder,
    setup_logging,
    get_logger,
    log_with_context
)
from src.exceptions import (
    ModelError,
    CacheError,
    ModelVersionError,
    DatabaseError,
    ValidationError
)

# Import existing test modules
from .test_logging_config import TestLoggingConfig
from .test_model_versioning import TestModelVersioning
from .test_model_caching import TestModelCaching
from .test_model_interface import TestModelInterface
from .test_ensemble_and_versioning import TestEnsembleAndVersioning
from .test_database_manager import TestDatabaseManager
from .test_plotting import TestPlotting
from .test_pdb_encoder import TestPDBEncoder
from .test_file_processing import TestFileProcessing

class TestUnit:
    """Unit tests for individual components."""
    
    def test_logging_config(self, temp_dir):
        """Test logging configuration."""
        # Set up logging
        setup_logging(log_dir=temp_dir)
        logger = get_logger('test')
        
        # Test logging with context
        with log_with_context(logger, {'test_id': '123'}):
            logger.info('Test message')
        
        # Verify log file exists
        log_files = list(Path(temp_dir).glob('*.log'))
        assert len(log_files) > 0
        
        # Verify log content
        log_content = log_files[0].read_text()
        assert 'Test message' in log_content
        assert 'test_id' in log_content
    
    def test_model_versioning(self, test_version_manager, test_model_weights, temp_dir):
        """Test model versioning system."""
        # Create test weights file
        weights_path = os.path.join(temp_dir, 'test_weights.pt')
        torch.save(test_model_weights, weights_path)
        
        # Add version
        version = test_version_manager.add_version(
            model_name='test_model',
            model_type='test_type',
            weights_path=weights_path,
            metadata={'version': '1.0.0'}
        )
        
        # Verify version
        assert version.model_name == 'test_model'
        assert version.model_type == 'test_type'
        assert version.metadata['version'] == '1.0.0'
        
        # Test version retrieval
        retrieved = test_version_manager.get_version(version.version_id)
        assert retrieved.version_id == version.version_id
    
    def test_model_caching(self, test_cache, test_model_weights, temp_dir):
        """Test model caching system."""
        # Create test weights file
        weights_path = os.path.join(temp_dir, 'test_weights.pt')
        torch.save(test_model_weights, weights_path)
        
        # Add to cache
        entry = test_cache.add(
            model_name='test_model',
            model_type='test_type',
            weights_path=weights_path,
            metadata={'version': '1.0.0'}
        )
        
        # Verify cache entry
        assert entry.model_name == 'test_model'
        assert entry.model_type == 'test_type'
        assert entry.metadata['version'] == '1.0.0'
        
        # Test cache retrieval
        retrieved = test_cache.get(entry.checksum)
        assert retrieved.checksum == entry.checksum
    
    def test_database_operations(self, test_database, test_config, test_metadata):
        """Test database operations."""
        # Create entry
        entry = DatabaseEntry(
            model_name=test_config['model_name'],
            model_type=test_config['model_type'],
            model_path=test_config['model_path'],
            metadata=test_metadata
        )
        
        # Add entry
        test_database.add_entry(entry)
        
        # Query entry
        result = test_database.get_entry(entry.id)
        
        # Verify result
        assert result.model_name == test_config['model_name']
        assert result.model_type == test_config['model_type']
        assert result.metadata == test_metadata
    
    def test_pdb_operations(self, test_pdb_file):
        """Test PDB file operations."""
        # Initialize encoder/decoder
        encoder = PDBEncoder()
        decoder = PDBDecoder()
        
        # Read PDB file
        structure = decoder.read_pdb(test_pdb_file)
        
        # Verify structure
        assert 'coordinates' in structure
        assert 'sequence' in structure
        assert 'confidence' in structure
        
        # Encode structure
        encoded = encoder.encode_structure(structure)
        
        # Verify encoding
        assert isinstance(encoded, dict)
        assert 'coordinate_tensor' in encoded
        assert 'sequence_tensor' in encoded
        assert 'confidence_tensor' in encoded

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_model_workflow(self, test_model_interface, test_sequence, temp_dir):
        """Test complete model workflow."""
        # Generate structure
        output_file = os.path.join(temp_dir, 'output.pdb')
        structure = test_model_interface.generate_structure(
            sequence=test_sequence,
            output_file=output_file
        )
        
        # Extract embeddings
        embeddings = test_model_interface.extract_embeddings(sequence=test_sequence)
        
        # Verify outputs
        assert os.path.exists(output_file)
        assert isinstance(embeddings, dict)
        assert 'structure_embedding' in embeddings
        assert 'pair_embedding' in embeddings
        assert 'msa_embedding' in embeddings
    
    def test_ensemble_workflow(self, temp_dir, test_ensemble_data):
        """Test ensemble analysis workflow."""
        # Initialize models
        models = [
            ModelInterface(
                model_name=f'test_model_{i}',
                model_type='test_type',
                model_path=os.path.join(temp_dir, f'model_{i}.pt')
            )
            for i in range(3)
        ]
        
        # Generate structures
        structures = []
        for i, model in enumerate(models):
            output_file = os.path.join(temp_dir, f'structure_{i}.pdb')
            structure = model.generate_structure(
                sequence='MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                output_file=output_file
            )
            structures.append(structure)
        
        # Perform ensemble analysis
        from src.utils import EnsembleAnalyzer
        analyzer = EnsembleAnalyzer()
        results = analyzer.analyze_structures(structures)
        
        # Verify results
        assert 'rmsd_matrix' in results
        assert 'consensus_structure' in results
        assert 'confidence_scores' in results
    
    def test_caching_workflow(self, test_cache, test_model_interface, test_sequence, temp_dir):
        """Test model caching workflow."""
        # Generate structure
        output_file = os.path.join(temp_dir, 'output.pdb')
        structure = test_model_interface.generate_structure(
            sequence=test_sequence,
            output_file=output_file
        )
        
        # Add to cache
        entry = test_cache.add(
            model_name=test_model_interface.model_name,
            model_type=test_model_interface.model_type,
            weights_path=test_model_interface.model_path,
            metadata={'version': '1.0.0'}
        )
        
        # Retrieve from cache
        cached_weights = test_cache.get(entry.checksum)
        
        # Verify cache entry
        assert cached_weights.model_name == test_model_interface.model_name
        assert cached_weights.model_type == test_model_interface.model_type
        assert cached_weights.metadata['version'] == '1.0.0'
    
    def test_versioning_workflow(self, test_version_manager, test_model_interface, test_sequence, temp_dir):
        """Test model versioning workflow."""
        # Generate structure
        output_file = os.path.join(temp_dir, 'output.pdb')
        structure = test_model_interface.generate_structure(
            sequence=test_sequence,
            output_file=output_file
        )
        
        # Add version
        version = test_version_manager.add_version(
            model_name=test_model_interface.model_name,
            model_type=test_model_interface.model_type,
            weights_path=test_model_interface.model_path,
            metadata={'version': '1.0.0'}
        )
        
        # Create child version
        child_version = test_version_manager.add_version(
            model_name=test_model_interface.model_name,
            model_type=test_model_interface.model_type,
            weights_path=test_model_interface.model_path,
            metadata={'version': '1.0.1'},
            parent_version=version.version_id
        )
        
        # Verify version hierarchy
        assert child_version.parent_version == version.version_id
        assert version.version_id in test_version_manager.get_version_history(child_version.version_id)
    
    def test_error_handling(self, temp_dir):
        """Test error handling workflow."""
        # Test invalid model path
        with pytest.raises(ModelError):
            ModelInterface(
                model_name='test_model',
                model_type='test_type',
                model_path='nonexistent.pt'
            )
        
        # Test invalid cache entry
        cache = ModelCache(cache_dir=temp_dir)
        with pytest.raises(CacheError):
            cache.get('nonexistent_checksum')
        
        # Test invalid version
        version_manager = ModelVersionManager(temp_dir)
        with pytest.raises(ModelVersionError):
            version_manager.get_version('nonexistent_version')
        
        # Test invalid database entry
        db = DatabaseManager(os.path.join(temp_dir, 'test.db'))
        with pytest.raises(DatabaseError):
            db.get_entry('nonexistent_id')
    
    def test_memory_management(self, test_model_interface):
        """Test memory management workflow."""
        # Optimize memory
        test_model_interface.optimize_memory_usage()
        
        # Quantize model
        test_model_interface.quantize_model()
        
        # Verify memory usage
        assert torch.cuda.memory_allocated() < torch.cuda.max_memory_allocated()

if __name__ == '__main__':
    pytest.main([__file__]) 