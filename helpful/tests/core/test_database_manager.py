import unittest
import tempfile
import shutil
from pathlib import Path
from multi_model_analysis.utils.database_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test DatabaseManager initialization."""
        self.assertEqual(self.db_manager.database_dir, Path(self.temp_dir))
        self.assertTrue(self.db_manager.database_dir.exists())
        self.assertTrue(self.db_manager.database_dir.is_dir())
    
    def test_database_configurations(self):
        """Test database configurations."""
        required_dbs = ['uniref90', 'mgnify', 'bfd', 'uniclust30', 'pdb70', 'pdb_mmcif']
        for db in required_dbs:
            self.assertIn(db, self.db_manager.databases)
            self.assertTrue(self.db_manager.databases[db]['required'])
            self.assertIn('url', self.db_manager.databases[db])
            self.assertIn('md5', self.db_manager.databases[db])
            self.assertIn('size', self.db_manager.databases[db])
    
    def test_verify_database(self):
        """Test database verification."""
        # Test with non-existent database
        self.assertFalse(self.db_manager.verify_database('uniref90'))
        
        # Test with invalid database name
        with self.assertRaises(ValueError):
            self.db_manager.verify_database('invalid_db')
    
    def test_get_database_path(self):
        """Test getting database path."""
        # Test with non-existent database
        self.assertIsNone(self.db_manager.get_database_path('uniref90'))
        
        # Test with invalid database name
        with self.assertRaises(ValueError):
            self.db_manager.get_database_path('invalid_db')
    
    def test_save_and_load_database_state(self):
        """Test saving and loading database state."""
        state_file = Path(self.temp_dir) / 'database_state.json'
        
        # Save state
        self.db_manager.save_database_state(str(state_file))
        self.assertTrue(state_file.exists())
        
        # Load state
        state = self.db_manager.load_database_state(str(state_file))
        self.assertIsInstance(state, dict)
        self.assertEqual(len(state), len(self.db_manager.databases))
        
        # Verify state structure
        for db_name in self.db_manager.databases:
            self.assertIn(db_name, state)
            self.assertIn('exists', state[db_name])
            self.assertIn('valid', state[db_name])
    
    def test_setup_all_databases(self):
        """Test setting up all databases."""
        # Note: This test will not actually download databases
        # as it would require significant time and bandwidth
        results = self.db_manager.setup_all_databases()
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(self.db_manager.databases))
        
        for db_name, success in results.items():
            self.assertIsInstance(success, bool)
            self.assertIn(db_name, self.db_manager.databases)

if __name__ == '__main__':
    unittest.main() 