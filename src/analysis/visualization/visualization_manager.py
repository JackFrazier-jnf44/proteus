"""Visualization manager for protein structure analysis."""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import Structure
import seaborn as sns
from dataclasses import dataclass

from src.exceptions import VisualizationError

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    output_dir: str
    dpi: int = 300
    figure_size: tuple = (10, 8)
    color_scheme: str = 'viridis'
    save_format: str = 'png'
    interactive: bool = False

class VisualizationManager:
    """Manages visualization tasks for protein structure analysis."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize visualization manager.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn')
        sns.set_palette(config.color_scheme)
        
        logger.info(f"Initialized visualization manager with output directory: {self.output_dir}")
    
    def plot_confidence_scores(
        self,
        confidence_scores: Dict[str, np.ndarray],
        output_name: str,
        title: Optional[str] = None
    ) -> None:
        """Plot confidence scores for structure prediction.
        
        Args:
            confidence_scores: Dictionary of confidence metrics
            output_name: Name for output file
            title: Optional plot title
        """
        try:
            fig, axes = plt.subplots(
                len(confidence_scores),
                1,
                figsize=self.config.figure_size,
                squeeze=False
            )
            fig.suptitle(title or "Confidence Scores")
            
            for i, (metric, scores) in enumerate(confidence_scores.items()):
                ax = axes[i, 0]
                sns.heatmap(
                    scores,
                    ax=ax,
                    cmap=self.config.color_scheme,
                    vmin=0,
                    vmax=1
                )
                ax.set_title(f"{metric} Scores")
                ax.set_xlabel("Residue")
                ax.set_ylabel("Residue")
            
            plt.tight_layout()
            self._save_figure(fig, output_name)
            
        except Exception as e:
            logger.error(f"Failed to plot confidence scores: {str(e)}")
            raise VisualizationError(f"Confidence score plotting failed: {str(e)}")
    
    def plot_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        output_name: str,
        title: Optional[str] = None
    ) -> None:
        """Plot distance matrix.
        
        Args:
            distance_matrix: Distance matrix array
            output_name: Name for output file
            title: Optional plot title
        """
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            sns.heatmap(
                distance_matrix,
                ax=ax,
                cmap=self.config.color_scheme,
                vmin=0,
                vmax=np.max(distance_matrix)
            )
            ax.set_title(title or "Distance Matrix")
            ax.set_xlabel("Residue")
            ax.set_ylabel("Residue")
            
            plt.tight_layout()
            self._save_figure(fig, output_name)
            
        except Exception as e:
            logger.error(f"Failed to plot distance matrix: {str(e)}")
            raise VisualizationError(f"Distance matrix plotting failed: {str(e)}")
    
    def plot_attention_maps(
        self,
        attention_maps: Dict[str, np.ndarray],
        output_name: str,
        title: Optional[str] = None
    ) -> None:
        """Plot attention maps.
        
        Args:
            attention_maps: Dictionary of attention maps
            output_name: Name for output file
            title: Optional plot title
        """
        try:
            fig, axes = plt.subplots(
                len(attention_maps),
                1,
                figsize=self.config.figure_size,
                squeeze=False
            )
            fig.suptitle(title or "Attention Maps")
            
            for i, (layer, attention) in enumerate(attention_maps.items()):
                ax = axes[i, 0]
                sns.heatmap(
                    attention,
                    ax=ax,
                    cmap=self.config.color_scheme,
                    vmin=0,
                    vmax=1
                )
                ax.set_title(f"{layer} Attention")
                ax.set_xlabel("Query")
                ax.set_ylabel("Key")
            
            plt.tight_layout()
            self._save_figure(fig, output_name)
            
        except Exception as e:
            logger.error(f"Failed to plot attention maps: {str(e)}")
            raise VisualizationError(f"Attention map plotting failed: {str(e)}")
    
    def plot_ensemble_comparison(
        self,
        structures: List[Structure],
        output_name: str,
        title: Optional[str] = None
    ) -> None:
        """Plot ensemble structure comparison.
        
        Args:
            structures: List of structures to compare
            output_name: Name for output file
            title: Optional plot title
        """
        try:
            # Calculate RMSD matrix
            rmsd_matrix = self._calculate_rmsd_matrix(structures)
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            sns.heatmap(
                rmsd_matrix,
                ax=ax,
                cmap=self.config.color_scheme,
                vmin=0,
                vmax=np.max(rmsd_matrix)
            )
            ax.set_title(title or "Ensemble RMSD Matrix")
            ax.set_xlabel("Model")
            ax.set_ylabel("Model")
            
            plt.tight_layout()
            self._save_figure(fig, output_name)
            
        except Exception as e:
            logger.error(f"Failed to plot ensemble comparison: {str(e)}")
            raise VisualizationError(f"Ensemble comparison plotting failed: {str(e)}")
    
    def _calculate_rmsd_matrix(self, structures: List[Structure]) -> np.ndarray:
        """Calculate RMSD matrix between structures.
        
        Args:
            structures: List of structures to compare
            
        Returns:
            RMSD matrix
        """
        n_structures = len(structures)
        rmsd_matrix = np.zeros((n_structures, n_structures))
        
        for i in range(n_structures):
            for j in range(i+1, n_structures):
                rmsd = self._calculate_rmsd(structures[i], structures[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        return rmsd_matrix
    
    def _calculate_rmsd(self, structure1: Structure, structure2: Structure) -> float:
        """Calculate RMSD between two structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            
        Returns:
            RMSD value
        """
        # Extract CA coordinates
        coords1 = []
        coords2 = []
        
        for model1, model2 in zip(structure1, structure2):
            for chain1, chain2 in zip(model1, model2):
                for residue1, residue2 in zip(chain1, chain2):
                    if 'CA' in residue1 and 'CA' in residue2:
                        coords1.append(residue1['CA'].get_coord())
                        coords2.append(residue2['CA'].get_coord())
        
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        
        # Calculate RMSD
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    
    def _save_figure(self, fig: plt.Figure, output_name: str) -> None:
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure
            output_name: Name for output file
        """
        try:
            output_path = self.output_dir / f"{output_name}.{self.config.save_format}"
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.debug(f"Saved figure to {output_path}")
            
            if self.config.interactive:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
            raise VisualizationError(f"Figure saving failed: {str(e)}") 