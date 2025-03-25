import unittest
from multi_model_analysis.utils.file_processing import read_pdb_file, write_pdb_file
from multi_model_analysis.utils.pdb_encoder import encode_pdb_to_distance_matrix

class TestFileProcessing(unittest.TestCase):

    def test_read_pdb_file(self):
        # Test reading a valid PDB file
        pdb_content = read_pdb_file('path/to/valid.pdb')
        self.assertIsInstance(pdb_content, str)
        self.assertIn('ATOM', pdb_content)

    def test_write_pdb_file(self):
        # Test writing to a PDB file
        content = "ATOM      1  N   MET A   1      20.154  34.123  27.456  1.00 20.00           N"
        write_pdb_file('path/to/output.pdb', content)
        with open('path/to/output.pdb', 'r') as f:
            written_content = f.read()
        self.assertEqual(content, written_content)

    def test_encode_pdb_to_distance_matrix(self):
        # Test encoding a PDB file to a distance matrix
        distance_matrix = encode_pdb_to_distance_matrix('path/to/valid.pdb')
        self.assertIsInstance(distance_matrix, np.ndarray)
        self.assertEqual(distance_matrix.shape[0], distance_matrix.shape[1])  # Should be square

if __name__ == '__main__':
    unittest.main()