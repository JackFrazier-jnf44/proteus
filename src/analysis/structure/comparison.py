"""Structure comparison module."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import three_to_one
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
import MDAnalysis.analysis.distance as dist

logger = logging.getLogger(__name__)

class StructureComparer:
    """Class for comparing protein structures using various metrics."""
    
    def __init__(self, parser: Optional[PDBParser] = None):
        """Initialize the structure comparer.
        
        Args:
            parser: Optional PDBParser instance
        """
        self.parser = parser or PDBParser(QUIET=True)
    
    def load_structure(self, structure_path: Union[str, Path]) -> Structure:
        """Load a structure from a file.
        
        Args:
            structure_path: Path to structure file
            
        Returns:
            Bio.PDB.Structure object
        """
        return self.parser.get_structure("structure", str(structure_path))
    
    def calculate_rmsd(
        self,
        structure1: Union[str, Path, Structure],
        structure2: Union[str, Path, Structure],
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Calculate RMSD between two structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            chain_id: Optional chain ID to compare
            residue_range: Optional residue range to compare
            
        Returns:
            RMSD value
        """
        if isinstance(structure1, (str, Path)):
            structure1 = self.load_structure(structure1)
        if isinstance(structure2, (str, Path)):
            structure2 = self.load_structure(structure2)
            
        # Get CA atoms
        atoms1 = self._get_ca_atoms(structure1, chain_id, residue_range)
        atoms2 = self._get_ca_atoms(structure2, chain_id, residue_range)
        
        if len(atoms1) != len(atoms2):
            raise ValueError("Structures have different numbers of residues")
            
        # Calculate RMSD
        coords1 = np.array([atom.get_coord() for atom in atoms1])
        coords2 = np.array([atom.get_coord() for atom in atoms2])
        
        # Superimpose structures
        sup = Superimposer()
        sup.set_atoms(atoms1, atoms2)
        rmsd = sup.rms
        
        return rmsd
    
    def calculate_tm_score(
        self,
        structure1: Union[str, Path, Structure],
        structure2: Union[str, Path, Structure],
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Calculate TM-score between two structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            chain_id: Optional chain ID to compare
            residue_range: Optional residue range to compare
            
        Returns:
            TM-score value
        """
        if isinstance(structure1, (str, Path)):
            structure1 = self.load_structure(structure1)
        if isinstance(structure2, (str, Path)):
            structure2 = self.load_structure(structure2)
            
        # Get CA atoms
        atoms1 = self._get_ca_atoms(structure1, chain_id, residue_range)
        atoms2 = self._get_ca_atoms(structure2, chain_id, residue_range)
        
        if len(atoms1) != len(atoms2):
            raise ValueError("Structures have different numbers of residues")
            
        # Calculate TM-score
        coords1 = np.array([atom.get_coord() for atom in atoms1])
        coords2 = np.array([atom.get_coord() for atom in atoms2])
        
        # Superimpose structures
        sup = Superimposer()
        sup.set_atoms(atoms1, atoms2)
        
        # Calculate TM-score
        d0 = 1.24 * (len(atoms1) - 15) ** (1/3) - 1.8
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        tm_score = np.mean(1 / (1 + (distances / d0) ** 2))
        
        return tm_score
    
    def calculate_contact_map_overlap(
        self,
        structure1: Union[str, Path, Structure],
        structure2: Union[str, Path, Structure],
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
        distance_threshold: float = 8.0,
    ) -> float:
        """Calculate contact map overlap between two structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            chain_id: Optional chain ID to compare
            residue_range: Optional residue range to compare
            distance_threshold: Distance threshold for contact definition
            
        Returns:
            Contact map overlap score
        """
        if isinstance(structure1, (str, Path)):
            structure1 = self.load_structure(structure1)
        if isinstance(structure2, (str, Path)):
            structure2 = self.load_structure(structure2)
            
        # Get CA atoms
        atoms1 = self._get_ca_atoms(structure1, chain_id, residue_range)
        atoms2 = self._get_ca_atoms(structure2, chain_id, residue_range)
        
        if len(atoms1) != len(atoms2):
            raise ValueError("Structures have different numbers of residues")
            
        # Calculate contact maps
        coords1 = np.array([atom.get_coord() for atom in atoms1])
        coords2 = np.array([atom.get_coord() for atom in atoms2])
        
        # Calculate distance matrices
        dist_matrix1 = np.sqrt(np.sum((coords1[:, np.newaxis] - coords1) ** 2, axis=2))
        dist_matrix2 = np.sqrt(np.sum((coords2[:, np.newaxis] - coords2) ** 2, axis=2))
        
        # Create contact maps
        contact_map1 = dist_matrix1 < distance_threshold
        contact_map2 = dist_matrix2 < distance_threshold
        
        # Calculate overlap
        overlap = np.sum(contact_map1 & contact_map2)
        total = np.sum(contact_map1 | contact_map2)
        
        return overlap / total if total > 0 else 0.0
    
    def calculate_secondary_structure_similarity(
        self,
        structure1: Union[str, Path, Structure],
        structure2: Union[str, Path, Structure],
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Calculate secondary structure similarity between two structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            chain_id: Optional chain ID to compare
            residue_range: Optional residue range to compare
            
        Returns:
            Secondary structure similarity score
        """
        if isinstance(structure1, (str, Path)):
            structure1 = self.load_structure(structure1)
        if isinstance(structure2, (str, Path)):
            structure2 = self.load_structure(structure2)
            
        # Calculate DSSP for both structures
        dssp1 = DSSP(structure1, structure1)
        dssp2 = DSSP(structure2, structure2)
        
        # Get secondary structure assignments
        ss1 = []
        ss2 = []
        
        for model in structure1:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue_range and not (residue_range[0] <= residue.id[1] <= residue_range[1]):
                        continue
                    if is_aa(residue):
                        ss1.append(dssp1[(model.id, chain.id, residue.id)][2])
                        
        for model in structure2:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue_range and not (residue_range[0] <= residue.id[1] <= residue_range[1]):
                        continue
                    if is_aa(residue):
                        ss2.append(dssp2[(model.id, chain.id, residue.id)][2])
        
        if len(ss1) != len(ss2):
            raise ValueError("Structures have different numbers of residues")
            
        # Calculate similarity
        matches = sum(1 for s1, s2 in zip(ss1, ss2) if s1 == s2)
        return matches / len(ss1)
    
    def calculate_dihedral_angle_similarity(
        self,
        structure1: Union[str, Path, Structure],
        structure2: Union[str, Path, Structure],
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Calculate dihedral angle similarity between two structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            chain_id: Optional chain ID to compare
            residue_range: Optional residue range to compare
            
        Returns:
            Dihedral angle similarity score
        """
        if isinstance(structure1, (str, Path)):
            structure1 = self.load_structure(structure1)
        if isinstance(structure2, (str, Path)):
            structure2 = self.load_structure(structure2)
            
        # Get dihedral angles
        angles1 = self._get_dihedral_angles(structure1, chain_id, residue_range)
        angles2 = self._get_dihedral_angles(structure2, chain_id, residue_range)
        
        if len(angles1) != len(angles2):
            raise ValueError("Structures have different numbers of residues")
            
        # Calculate similarity
        angle_diffs = np.abs(np.array(angles1) - np.array(angles2))
        angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
        similarity = 1 - np.mean(angle_diffs) / 180
        
        return similarity
    
    def _get_ca_atoms(
        self,
        structure: Structure,
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
    ) -> List:
        """Get CA atoms from a structure.
        
        Args:
            structure: Bio.PDB.Structure object
            chain_id: Optional chain ID to filter
            residue_range: Optional residue range to filter
            
        Returns:
            List of CA atoms
        """
        atoms = []
        for model in structure:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue_range and not (residue_range[0] <= residue.id[1] <= residue_range[1]):
                        continue
                    if is_aa(residue):
                        if "CA" in residue:
                            atoms.append(residue["CA"])
        return atoms
    
    def _get_dihedral_angles(
        self,
        structure: Structure,
        chain_id: Optional[str] = None,
        residue_range: Optional[Tuple[int, int]] = None,
    ) -> List[float]:
        """Get dihedral angles from a structure.
        
        Args:
            structure: Bio.PDB.Structure object
            chain_id: Optional chain ID to filter
            residue_range: Optional residue range to filter
            
        Returns:
            List of dihedral angles
        """
        angles = []
        for model in structure:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue_range and not (residue_range[0] <= residue.id[1] <= residue_range[1]):
                        continue
                    if is_aa(residue):
                        if all(atom in residue for atom in ["N", "CA", "C"]):
                            angle = calc_dihedral(
                                residue["N"].get_vector(),
                                residue["CA"].get_vector(),
                                residue["C"].get_vector(),
                            )
                            angles.append(angle)
        return angles

def compare_structures(
    structure1_file: str,
    structure2_file: str,
    chain_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two protein structures and calculate relevant metrics.
    
    Args:
        structure1_file: Path to first structure file (PDB format)
        structure2_file: Path to second structure file (PDB format)
        chain_id: Optional chain ID to compare (if None, compares all chains)
        
    Returns:
        Dictionary containing comparison metrics:
        - rmsd: Root mean square deviation
        - gdt_ts: Global distance test (total score)
        - gdt_ha: Global distance test (high accuracy)
        - tm_score: Template modeling score
        - lddt: Local distance difference test score
        - contact_map_correlation: Correlation between contact maps
    """
    try:
        logger.info(f"Comparing structures: {structure1_file} and {structure2_file}")
        
        # Load structures using MDAnalysis
        universe1 = mda.Universe(structure1_file)
        universe2 = mda.Universe(structure2_file)
        
        # Select atoms for comparison
        if chain_id:
            selection1 = universe1.select_atoms(f"protein and segid {chain_id} and name CA")
            selection2 = universe2.select_atoms(f"protein and segid {chain_id} and name CA")
        else:
            selection1 = universe1.select_atoms("protein and name CA")
            selection2 = universe2.select_atoms("protein and name CA")
        
        if len(selection1) != len(selection2):
            raise ValueError("Structures have different numbers of CA atoms")
        
        # Calculate RMSD
        rmsd = rms.rmsd(selection1.positions, selection2.positions)
        
        # Calculate distance matrices
        dist_matrix1 = dist.distance_array(selection1.positions, selection1.positions)
        dist_matrix2 = dist.distance_array(selection2.positions, selection2.positions)
        
        # Calculate contact maps (8Å threshold)
        contact_map1 = dist_matrix1 < 8.0
        contact_map2 = dist_matrix2 < 8.0
        
        # Calculate contact map correlation
        contact_correlation = np.corrcoef(contact_map1.flatten(), contact_map2.flatten())[0,1]
        
        # Calculate GDT scores
        gdt_thresholds = [1.0, 2.0, 4.0, 8.0]  # Angstroms
        gdt_counts = []
        
        for threshold in gdt_thresholds:
            distances = np.sqrt(np.sum((selection1.positions - selection2.positions)**2, axis=1))
            count = np.sum(distances <= threshold)
            gdt_counts.append(count / len(selection1))
        
        gdt_ts = sum(gdt_counts) * 25  # GDT_TS score
        
        # Calculate GDT_HA score (using 0.5, 1, 2, 4 Angstrom thresholds)
        gdt_ha_thresholds = [0.5, 1.0, 2.0, 4.0]
        gdt_ha_counts = []
        
        for threshold in gdt_ha_thresholds:
            distances = np.sqrt(np.sum((selection1.positions - selection2.positions)**2, axis=1))
            count = np.sum(distances <= threshold)
            gdt_ha_counts.append(count / len(selection1))
        
        gdt_ha = sum(gdt_ha_counts) * 25  # GDT_HA score
        
        # Calculate TM-score
        d0 = 1.24 * (len(selection1) - 15) ** (1/3) - 1.8  # TM-score normalization factor
        distances = np.sqrt(np.sum((selection1.positions - selection2.positions)**2, axis=1))
        tm_score = np.mean(1 / (1 + (distances/d0)**2))
        
        # Calculate lDDT score
        lddt_thresholds = [0.5, 1.0, 2.0, 4.0]
        lddt_scores = []
        
        for threshold in lddt_thresholds:
            preserved_contacts = np.logical_and(
                dist_matrix1 < threshold,
                dist_matrix2 < threshold
            )
            total_contacts = np.sum(dist_matrix1 < threshold)
            if total_contacts > 0:
                lddt_scores.append(np.sum(preserved_contacts) / total_contacts)
            else:
                lddt_scores.append(0)
        
        lddt = np.mean(lddt_scores) * 100  # Scale to 0-100
        
        # Compile results
        results = {
            'rmsd': float(rmsd),
            'gdt_ts': float(gdt_ts),
            'gdt_ha': float(gdt_ha),
            'tm_score': float(tm_score),
            'lddt': float(lddt),
            'contact_map_correlation': float(contact_correlation)
        }
        
        logger.info("Structure comparison completed successfully")
        logger.debug(f"Comparison results: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to compare structures: {str(e)}")
        raise 