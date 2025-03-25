"""Dataset management for benchmarking."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from Bio import SeqIO
from Bio.PDB import PDBParser, Structure

@dataclass
class ProteinEntry:
    """A single protein entry in a dataset."""
    
    id: str  # Unique identifier
    sequence: str  # Amino acid sequence
    structure: Optional[Structure]  # PDB structure if available
    metadata: Dict[str, any]  # Additional metadata

@dataclass
class BenchmarkDataset:
    """Base class for benchmark datasets."""
    
    name: str
    description: str
    entries: List[ProteinEntry]
    metadata: Dict[str, any]
    
    @classmethod
    def from_directory(cls, path: Union[str, Path], pattern: str = "*.pdb") -> 'BenchmarkDataset':
        """Load dataset from a directory of PDB files."""
        path = Path(path)
        entries = []
        parser = PDBParser(QUIET=True)
        
        for pdb_file in path.glob(pattern):
            try:
                # Load structure
                structure = parser.get_structure(pdb_file.stem, pdb_file)
                
                # Extract sequence from structure
                sequence = ""
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if 'CA' in residue:
                                sequence += residue.resname
                
                # Create entry
                entry = ProteinEntry(
                    id=pdb_file.stem,
                    sequence=sequence,
                    structure=structure,
                    metadata={"source_file": str(pdb_file)}
                )
                entries.append(entry)
                
            except Exception as e:
                print(f"Error loading {pdb_file}: {e}")
        
        return cls(
            name=path.name,
            description=f"Dataset loaded from {path}",
            entries=entries,
            metadata={"source_dir": str(path)}
        )
    
    @classmethod
    def from_fasta(cls, fasta_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None) -> 'BenchmarkDataset':
        """Load dataset from a FASTA file with optional metadata JSON."""
        fasta_path = Path(fasta_path)
        entries = []
        
        # Load metadata if provided
        metadata = {}
        if metadata_path:
            with open(metadata_path) as f:
                metadata = json.load(f)
        
        # Load sequences
        for record in SeqIO.parse(fasta_path, "fasta"):
            entry = ProteinEntry(
                id=record.id,
                sequence=str(record.seq),
                structure=None,
                metadata=metadata.get(record.id, {})
            )
            entries.append(entry)
        
        return cls(
            name=fasta_path.stem,
            description=f"Dataset loaded from {fasta_path}",
            entries=entries,
            metadata={"source_file": str(fasta_path)}
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save sequences to FASTA
        with open(path / f"{self.name}.fasta", "w") as f:
            for entry in self.entries:
                f.write(f">{entry.id}\n{entry.sequence}\n")
        
        # Save structures to PDB files
        for entry in self.entries:
            if entry.structure:
                pdb_path = path / f"{entry.id}.pdb"
                # Implementation would use Bio.PDB.PDBIO to save structure
        
        # Save metadata
        metadata = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "entries": {
                entry.id: entry.metadata for entry in self.entries
            }
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

class ProteinDataset(BenchmarkDataset):
    """Dataset specifically for protein structure prediction benchmarking."""
    
    def get_sequence_lengths(self) -> List[int]:
        """Get list of sequence lengths in the dataset."""
        return [len(entry.sequence) for entry in self.entries]
    
    def get_length_distribution(self) -> Dict[str, float]:
        """Get statistics about sequence length distribution."""
        lengths = self.get_sequence_lengths()
        return {
            "min": float(min(lengths)),
            "max": float(max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths))
        }
    
    def filter_by_length(self, min_length: int = 0, max_length: Optional[int] = None) -> 'ProteinDataset':
        """Create new dataset filtered by sequence length."""
        filtered_entries = [
            entry for entry in self.entries
            if len(entry.sequence) >= min_length and
            (max_length is None or len(entry.sequence) <= max_length)
        ]
        
        return ProteinDataset(
            name=f"{self.name}_filtered",
            description=f"{self.description} (length filtered)",
            entries=filtered_entries,
            metadata={
                **self.metadata,
                "length_filter": {"min": min_length, "max": max_length}
            }
        )
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
             random_seed: Optional[int] = None) -> Dict[str, 'ProteinDataset']:
        """Split dataset into train/validation/test sets."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        indices = np.random.permutation(len(self.entries))
        train_size = int(len(indices) * train_ratio)
        val_size = int(len(indices) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        def create_subset(name: str, idx: np.ndarray) -> ProteinDataset:
            return ProteinDataset(
                name=f"{self.name}_{name}",
                description=f"{self.description} ({name} split)",
                entries=[self.entries[i] for i in idx],
                metadata={**self.metadata, "split": name}
            )
        
        return {
            "train": create_subset("train", train_indices),
            "val": create_subset("val", val_indices),
            "test": create_subset("test", test_indices)
        } 