import unittest
from src.utils.pdb_encoder import encode_pdb_to_distance_matrix

class TestPDBEncoder(unittest.TestCase):

    def test_encode_pdb_to_distance_matrix_valid(self):
        # Test with a valid PDB file
        pdb_file = 'path/to/valid.pdb'
        expected_output_shape = (number_of_amino_acids, number_of_amino_acids)  # Replace with actual expected shape
        distance_matrix = encode_pdb_to_distance_matrix(pdb_file)
        self.assertEqual(distance_matrix.shape, expected_output_shape)

    def test_encode_pdb_to_distance_matrix_invalid(self):
        # Test with an invalid PDB file
        pdb_file = 'path/to/invalid.pdb'
        with self.assertRaises(ValueError):
            encode_pdb_to_distance_matrix(pdb_file)

    def test_encode_pdb_to_distance_matrix_empty(self):
        # Test with an empty PDB file
        pdb_file = 'path/to/empty.pdb'
        with self.assertRaises(ValueError):
            encode_pdb_to_distance_matrix(pdb_file)

if __name__ == '__main__':
    unittest.main()