import os
import logging
from pathlib import Path
import subprocess
from typing import Dict, List, Optional
import hashlib
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages AlphaFold database downloads and verification.
    
    This class handles downloading, verifying, and managing the required databases
    for AlphaFold structure prediction. It supports downloading from official sources
    and verifying database integrity.
    """
    
    def __init__(self, database_dir: str):
        """
        Initialize the database manager.
        
        Args:
            database_dir: Directory to store databases
        """
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Database configurations
        self.databases = {
            'uniref90': {
                'url': 'https://storage.googleapis.com/alphafold-databases/casp14_versions/uniref90_2021_03.fasta.gz',
                'md5': '8a1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1',  # Example MD5
                'size': 1000000000,  # Example size in bytes
                'required': True
            },
            'mgnify': {
                'url': 'https://storage.googleapis.com/alphafold-databases/casp14_versions/mgy_clusters_2022_05.fa.gz',
                'md5': '8b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1',
                'size': 2000000000,
                'required': True
            },
            'bfd': {
                'url': 'https://storage.googleapis.com/alphafold-databases/casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz',
                'md5': '8c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1',
                'size': 3000000000,
                'required': True
            },
            'uniclust30': {
                'url': 'https://storage.googleapis.com/alphafold-databases/casp14_versions/uniclust30_2018_08.tar.gz',
                'md5': '8d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1',
                'size': 4000000000,
                'required': True
            },
            'pdb70': {
                'url': 'https://storage.googleapis.com/alphafold-databases/casp14_versions/pdb70.tar.gz',
                'md5': '8e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1',
                'size': 5000000000,
                'required': True
            },
            'pdb_mmcif': {
                'url': 'https://storage.googleapis.com/alphafold-databases/casp14_versions/pdb_mmcif.tar.gz',
                'md5': '8f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1',
                'size': 6000000000,
                'required': True
            }
        }
    
    def verify_database(self, database_name: str) -> bool:
        """
        Verify the integrity of a downloaded database.
        
        Args:
            database_name: Name of the database to verify
            
        Returns:
            bool: True if database is valid, False otherwise
        """
        if database_name not in self.databases:
            raise ValueError(f"Unknown database: {database_name}")
        
        database_path = self.database_dir / database_name
        if not database_path.exists():
            return False
        
        # Check file size
        if database_path.stat().st_size != self.databases[database_name]['size']:
            logger.warning(f"Database {database_name} size mismatch")
            return False
        
        # Check MD5 hash
        with open(database_path, 'rb') as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
        
        if md5_hash != self.databases[database_name]['md5']:
            logger.warning(f"Database {database_name} MD5 mismatch")
            return False
        
        return True
    
    def download_database(self, database_name: str, force: bool = False) -> bool:
        """
        Download a database from the official source.
        
        Args:
            database_name: Name of the database to download
            force: Whether to force re-download even if database exists
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if database_name not in self.databases:
            raise ValueError(f"Unknown database: {database_name}")
        
        database_path = self.database_dir / database_name
        if database_path.exists() and not force:
            if self.verify_database(database_name):
                logger.info(f"Database {database_name} already exists and is valid")
                return True
            else:
                logger.warning(f"Existing database {database_name} is invalid, re-downloading")
        
        try:
            logger.info(f"Downloading database {database_name}")
            subprocess.run([
                'wget',
                '-O', str(database_path),
                self.databases[database_name]['url']
            ], check=True)
            
            if self.verify_database(database_name):
                logger.info(f"Successfully downloaded and verified {database_name}")
                return True
            else:
                logger.error(f"Downloaded database {database_name} verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download database {database_name}: {str(e)}")
            return False
    
    def setup_all_databases(self, force: bool = False) -> Dict[str, bool]:
        """
        Download and verify all required databases.
        
        Args:
            force: Whether to force re-download of existing databases
            
        Returns:
            Dict mapping database names to their setup status
        """
        results = {}
        for database_name, config in self.databases.items():
            if config['required']:
                results[database_name] = self.download_database(database_name, force)
        return results
    
    def get_database_path(self, database_name: str) -> Optional[Path]:
        """
        Get the path to a database.
        
        Args:
            database_name: Name of the database
            
        Returns:
            Path to the database if it exists and is valid, None otherwise
        """
        if database_name not in self.databases:
            raise ValueError(f"Unknown database: {database_name}")
        
        database_path = self.database_dir / database_name
        if database_path.exists() and self.verify_database(database_name):
            return database_path
        return None
    
    def save_database_state(self, state_file: str) -> None:
        """
        Save the current state of all databases to a file.
        
        Args:
            state_file: Path to save the state file
        """
        state = {
            'database_dir': str(self.database_dir),
            'databases': {
                name: {
                    'exists': (self.database_dir / name).exists(),
                    'valid': self.verify_database(name) if (self.database_dir / name).exists() else False
                }
                for name in self.databases
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_database_state(self, state_file: str) -> Dict[str, Dict[str, bool]]:
        """
        Load the state of databases from a file.
        
        Args:
            state_file: Path to the state file
            
        Returns:
            Dict mapping database names to their state
        """
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        if state['database_dir'] != str(self.database_dir):
            logger.warning("Database directory mismatch in state file")
        
        return state['databases'] 