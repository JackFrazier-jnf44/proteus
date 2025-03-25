"""PDB structure encoding module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.PDB.PDBIO import PDBIO

from ...config import SUPPORTED_FORMATS, ANALYSIS_SETTINGS

logger = logging.getLogger(__name__)

class PDBEncoder:
    """Class for encoding PDB structures into various formats."""
    
    def __init__(self):
        """Initialize the PDB encoder."""
        self.parser = PDBParser()
        self.mmcif_parser = MMCIFParser()
    
    def encode_to_distance_matrix(self, pdb_file: str) -> np.ndarray:
        """
        Convert a PDB file to a distance matrix.
        
        Args:
            pdb_file: Path to the PDB file
            
        Returns:
            Distance matrix as a numpy array
        """
        structure = self.parser.get_structure('protein', pdb_file)
        
        # Get all CA atoms
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
        
        # Calculate distance matrix
        n_atoms = len(ca_atoms)
        distance_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                distance_matrix[i, j] = ca_atoms[i] - ca_atoms[j]
        
        return distance_matrix
    
    def process_structure_files(self, model_outputs: Dict) -> Dict[str, np.ndarray]:
        """
        Process structure files into distance matrices.
        
        Args:
            model_outputs: Dictionary of model outputs containing structure files
            
        Returns:
            Dictionary mapping model names to their distance matrices
        """
        distance_matrices = {}
        for model_name, outputs in model_outputs.items():
            structure_file = outputs['structure_file']
            if structure_file.endswith(tuple(SUPPORTED_FORMATS['structure'])):
                try:
                    distance_matrices[model_name] = self.encode_to_distance_matrix(structure_file)
                    logger.debug(f"Distance matrix generated for {model_name}")
                except Exception as e:
                    logger.error(f"Error processing structure file for {model_name}: {str(e)}")
            else:
                logger.warning(f"Unsupported file format for {model_name}: {structure_file}")
        return distance_matrices
    
    def get_coordinates(self, structure) -> List[Tuple[float, float, float]]:
        """
        Extract coordinates from structure.
        
        Args:
            structure: Bio.PDB Structure object
            
        Returns:
            List of (x, y, z) coordinates
        """
        try:
            coordinates = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':  # Standard amino acid
                            if 'CA' in residue:
                                coordinates.append(residue['CA'].get_coord())
                            else:
                                logger.warning(f"No CA atom found in residue {residue.id}")
            return coordinates
        except Exception as e:
            logger.error(f"Failed to extract coordinates: {str(e)}")
            raise
    
    def calculate_distance_matrix(self, coordinates: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Calculate distance matrix from coordinates.
        
        Args:
            coordinates: List of (x, y, z) coordinates
            
        Returns:
            Distance matrix as numpy array
        """
        try:
            n_residues = len(coordinates)
            distance_matrix = np.zeros((n_residues, n_residues))
            
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    dist = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            return distance_matrix
        except Exception as e:
            logger.error(f"Failed to calculate distance matrix: {str(e)}")
            raise
    
    def validate_structure(self, structure) -> bool:
        """
        Validate protein structure.
        
        Args:
            structure: Bio.PDB Structure object
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check for models
            if len(structure) == 0:
                logger.warning("Structure has no models")
                return False
            
            # Check for chains
            for model in structure:
                if len(model) == 0:
                    logger.warning("Model has no chains")
                    return False
            
            # Check for residues
            for model in structure:
                for chain in model:
                    if len(chain) == 0:
                        logger.warning("Chain has no residues")
                        return False
            
            # Check for CA atoms
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ' and 'CA' not in residue:
                            logger.warning(f"Residue {residue.id} missing CA atom")
                            return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate structure: {str(e)}")
            return False 