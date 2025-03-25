import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from multi_model_analysis.models.model_interface import ModelInterface, ModelConfig
from multi_model_analysis.utils.ensemble import EnsembleConfig, EnsembleMethod
from multi_model_analysis.utils.model_versioning import ModelVersionManager

class TestEnsembleAndVersioning(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
        
        # Create dummy model paths and configs
        self.model_dirs = {}
        self.model_configs = []
        
        for model_name in ['alphafold', 'rosettafold']:
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
                config_path=str(model_dir / "config.yaml")
            )
            self.model_configs.append(config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_ensemble_initialization(self):
        """Test ensemble predictor initialization."""
        ensemble_config = EnsembleConfig(
            method=EnsembleMethod.AVERAGE,
            confidence_threshold=0.7
        )
        
        interface = ModelInterface(self.model_configs, ensemble_config)
        self.assertIsNotNone(interface.ensemble_predictor)
        self.assertEqual(interface.ensemble_predictor.config.method, EnsembleMethod.AVERAGE)
    
    def test_model_versioning(self):
        """Test model version management."""
        interface = ModelInterface(self.model_configs)
        
        # Test adding a new version
        for model_name in self.model_dirs:
            interface.add_model_version(
                model_name=f"{model_name}_test",
                version="1.0.0",
                model_path=self.model_dirs[model_name] / "weights",
                metadata={"accuracy": 0.95},
                dependencies={"torch": "1.9.0"}
            )
        
        # Test getting versions
        for model_name in self.model_dirs:
            versions = interface.get_model_versions(f"{model_name}_test")
            self.assertEqual(len(versions), 1)
            self.assertEqual(versions[0], "1.0.0")
        
        # Test getting latest version
        for model_name in self.model_dirs:
            latest = interface.get_latest_version(f"{model_name}_test")
            self.assertEqual(latest, "1.0.0")
        
        # Test switching versions
        for model_name in self.model_dirs:
            interface.switch_model_version(f"{model_name}_test", "1.0.0")
    
    def test_ensemble_predictions(self):
        """Test ensemble prediction generation."""
        ensemble_config = EnsembleConfig(
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            weights={"alphafold_test": 0.6, "rosettafold_test": 0.4},
            confidence_threshold=0.7
        )
        
        interface = ModelInterface(self.model_configs, ensemble_config)
        
        # Generate predictions
        outputs = interface.invoke_models(self.test_sequence, str(self.temp_dir))
        
        # Check individual model outputs
        self.assertIn("alphafold_test", outputs)
        self.assertIn("rosettafold_test", outputs)
        
        # Check ensemble outputs
        self.assertIn("ensemble", outputs)
        ensemble_embeddings = outputs["ensemble"]["embeddings"]
        
        # Verify ensemble embedding dimensions
        for key in ["structure_embedding", "pair_embedding", "msa_embedding"]:
            self.assertIn(key, ensemble_embeddings)
            self.assertEqual(ensemble_embeddings[key].shape[1], 1024)
    
    def test_ensemble_methods(self):
        """Test different ensemble methods."""
        methods = [
            EnsembleMethod.AVERAGE,
            EnsembleMethod.WEIGHTED_AVERAGE,
            EnsembleMethod.VOTING
        ]
        
        for method in methods:
            ensemble_config = EnsembleConfig(
                method=method,
                confidence_threshold=0.7
            )
            
            interface = ModelInterface(self.model_configs, ensemble_config)
            outputs = interface.invoke_models(self.test_sequence, str(self.temp_dir))
            
            self.assertIn("ensemble", outputs)
            ensemble_embeddings = outputs["ensemble"]["embeddings"]
            
            # Verify ensemble embedding dimensions
            for key in ["structure_embedding", "pair_embedding", "msa_embedding"]:
                self.assertIn(key, ensemble_embeddings)
                self.assertEqual(ensemble_embeddings[key].shape[1], 1024)

if __name__ == '__main__':
    unittest.main() 