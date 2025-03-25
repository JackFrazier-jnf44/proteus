import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import torch
from unittest.mock import Mock, patch
from src.interfaces.model_interface import (
    AlphaFoldInterface,
    RoseTTAFoldInterface,
    ModelConfig,
    ModelInterface,
    OpenFoldInterface,
    ESMInterface
)
from proteus.interfaces import BaseModelConfig
from proteus.exceptions import ModelError

class TestModelInterfaces(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
        
        # Create dummy model paths and configs
        self.model_dirs = {}
        self.model_configs = []
        
        for model_name in ['alphafold', 'rosettafold', 'openfold', 'esm']:
            model_dir = Path(self.temp_dir) / model_name
            model_dir.mkdir()
            self.model_dirs[model_name] = model_dir
            
            # Create dummy model files
            (model_dir / "weights").touch()
            (model_dir / "config.yaml").touch()
            
            # Create model config
            config = ModelConfig(
                name=f"{model_name}_test",
                model_type=model_name,
                output_format="pdb",
                embedding_config={
                    "structure_embedding": {"dimension": 1024},
                    "pair_embedding": {"dimension": 512},
                    "msa_embedding": {"dimension": 256}
                },
                model_path=str(model_dir / "weights"),
                config_path=str(model_dir / "config.yaml"),
                model_name="esm2_t36_3B_UR50D" if model_name == 'esm' else None
            )
            self.model_configs.append(config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('torch.load')
    def test_openfold_interface(self, mock_load):
        """Test OpenFold interface initialization and methods."""
        # Mock model and data transforms
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_load.return_value = {}
        
        with patch('proteus.models.model_interface.model.AlphaFold') as mock_alphafold:
            mock_alphafold.return_value = mock_model
            
            # Initialize interface
            interface = OpenFoldInterface(
                model_path=str(self.model_dirs['openfold'] / "weights"),
                config_path=str(self.model_dirs['openfold'] / "config.yaml")
            )
            
            # Test structure generation
            output_file = str(self.model_dirs['openfold'] / "output.pdb")
            interface.generate_structure(self.test_sequence, output_file)
            
            # Test embedding extraction
            embeddings = interface.extract_embeddings(self.test_sequence)
            self.assertIn('structure_embedding', embeddings)
            self.assertIn('pair_embedding', embeddings)
            self.assertIn('msa_embedding', embeddings)
    
    @patch('esm.pretrained.load_model_and_alphabet')
    def test_esm_interface(self, mock_load):
        """Test ESM interface initialization and methods."""
        # Mock model and alphabet
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.num_layers = 36
        mock_alphabet = Mock()
        mock_load.return_value = (mock_model, mock_alphabet)
        
        # Initialize interface
        interface = ESMInterface(model_name="esm2_t36_3B_UR50D")
        
        # Test structure generation
        output_file = str(self.model_dirs['esm'] / "output.pdb")
        interface.generate_structure(self.test_sequence, output_file)
        
        # Test embedding extraction
        embeddings = interface.extract_embeddings(self.test_sequence)
        self.assertIn('last_hidden', embeddings)
        self.assertIn('pooled', embeddings)
        self.assertIn('attention', embeddings)
    
    @patch('torch.load')
    def test_alphafold_interface(self, mock_load):
        """Test AlphaFold interface initialization and methods."""
        # Mock model and data pipeline
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_load.return_value = {}
        mock_pipeline = Mock()
        
        with patch('proteus.models.model_interface.alphafold_model.AlphaFold') as mock_alphafold, \
             patch('proteus.models.model_interface.pipeline.DataPipeline') as mock_pipeline_class:
            mock_alphafold.return_value = mock_model
            mock_pipeline_class.return_value = mock_pipeline
            
            # Initialize interface
            interface = AlphaFoldInterface(
                model_path=str(self.model_dirs['alphafold'] / "weights"),
                config_path=str(self.model_dirs['alphafold'] / "config.yaml")
            )
            
            # Test structure generation
            output_file = str(self.model_dirs['alphafold'] / "output.pdb")
            interface.generate_structure(self.test_sequence, output_file)
            
            # Test embedding extraction
            embeddings = interface.extract_embeddings(self.test_sequence)
            self.assertIn('structure_embedding', embeddings)
            self.assertIn('pair_embedding', embeddings)
            self.assertIn('msa_embedding', embeddings)
    
    @patch('torch.load')
    def test_rosettafold_interface(self, mock_load):
        """Test RoseTTAFold interface initialization and methods."""
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_load.return_value = {}
        
        with patch('proteus.models.model_interface.RoseTTAFold') as mock_rosettafold:
            mock_rosettafold.return_value = mock_model
            
            # Initialize interface
            interface = RoseTTAFoldInterface(
                model_path=str(self.model_dirs['rosettafold'] / "weights"),
                config_path=str(self.model_dirs['rosettafold'] / "config.yaml")
            )
            
            # Test structure generation
            output_file = str(self.model_dirs['rosettafold'] / "output.pdb")
            interface.generate_structure(self.test_sequence, output_file)
            
            # Test embedding extraction
            embeddings = interface.extract_embeddings(self.test_sequence)
            self.assertIn('structure_embedding', embeddings)
            self.assertIn('pair_embedding', embeddings)
            self.assertIn('msa_embedding', embeddings)
    
    def test_model_interface_with_all_models(self):
        """Test ModelInterface with all model types."""
        try:
            interface = ModelInterface(self.model_configs)
            
            # Test model initialization
            for model_name in ['alphafold', 'rosettafold', 'openfold', 'esm']:
                self.assertIn(f"{model_name}_test", interface.model_instances)
            
            # Test model invocation
            outputs = interface.invoke_models(self.test_sequence, str(self.temp_dir))
            
            # Verify outputs
            for model_name in ['alphafold', 'rosettafold', 'openfold', 'esm']:
                self.assertIn(f"{model_name}_test", outputs)
                self.assertIn('structure_file', outputs[f"{model_name}_test"])
                self.assertIn('embeddings', outputs[f"{model_name}_test"])
                
        except Exception as e:
            self.fail(f"ModelInterface test failed: {str(e)}")
    
    def test_embedding_dimensions(self):
        """Test embedding dimensions for all models."""
        interface = ModelInterface(self.model_configs)
        
        # Test AlphaFold embeddings
        alphafold_embeddings = interface.model_instances["alphafold_test"].extract_embeddings(self.test_sequence)
        self.assertEqual(alphafold_embeddings["structure_embedding"].shape[1], 1024)
        self.assertEqual(alphafold_embeddings["pair_embedding"].shape[1], 512)
        self.assertEqual(alphafold_embeddings["msa_embedding"].shape[1], 256)
        
        # Test RoseTTAFold embeddings
        rosettafold_embeddings = interface.model_instances["rosettafold_test"].extract_embeddings(self.test_sequence)
        self.assertEqual(rosettafold_embeddings["structure_embedding"].shape[1], 768)
        self.assertEqual(rosettafold_embeddings["pair_embedding"].shape[1], 384)
        self.assertEqual(rosettafold_embeddings["msa_embedding"].shape[1], 192)
        
        # Test OpenFold embeddings
        openfold_embeddings = interface.model_instances["openfold_test"].extract_embeddings(self.test_sequence)
        self.assertEqual(openfold_embeddings["structure_embedding"].shape[1], 512)
        self.assertEqual(openfold_embeddings["pair_embedding"].shape[1], 256)
        
        # Test ESM embeddings
        esm_embeddings = interface.model_instances["esm_test"].extract_embeddings(self.test_sequence)
        self.assertEqual(esm_embeddings["last_hidden"].shape[1], 1280)
        self.assertEqual(esm_embeddings["pooled"].shape[1], 1280)

class TestModelInterface:
    def test_model_initialization(self):
        """Test model interface initialization with different configurations."""
        config = BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm",
            output_format="pdb"
        )
        interface = ModelInterface(config)
        assert interface.model is not None
        assert interface.model_type == "esm"

    def test_structure_prediction(self):
        """Test basic structure prediction functionality."""
        config = BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        interface = ModelInterface(config)
        sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
        result = interface.predict_structure(sequence)
        
        assert "structure" in result
        assert "confidence" in result
        assert result["structure"].endswith(".pdb")

    def test_invalid_sequence(self):
        """Test handling of invalid input sequences."""
        config = BaseModelConfig(
            name="esm2_t33_650M_UR50D",
            model_type="esm"
        )
        interface = ModelInterface(config)
        
        with pytest.raises(ValueError):
            interface.predict_structure("NOT-A-VALID-SEQUENCE123")

if __name__ == '__main__':
    unittest.main()