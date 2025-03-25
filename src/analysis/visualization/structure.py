"""Structure visualization module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import *
from MDAnalysis import Universe
from MDAnalysis.analysis import rms

from .base import BaseVisualizer, VisualizationConfig
from ..structure.comparison import StructureComparer

logger = logging.getLogger(__name__)

class StructureVisualizer(BaseVisualizer):
    """Specialized visualizer for protein structure analysis."""
    
    def plot_contact_map(
        self,
        distance_matrix: np.ndarray,
        sequence: str,
        output_file: str,
        threshold: float = 8.0,
        title: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """Plot contact map from distance matrix."""
        try:
            logger.debug("Plotting contact map")
            
            # Create contact map
            contact_map = distance_matrix < threshold
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Plot contact map
            sns.heatmap(
                contact_map,
                cmap='binary',
                xticklabels=sequence,
                yticklabels=sequence,
                ax=ax
            )
            
            # Customize plot
            plot_title = title or "Contact Map"
            if model_name:
                plot_title = self._add_version_info_to_title(plot_title, model_name)
            ax.set_title(plot_title)
            ax.set_xlabel("Residue")
            ax.set_ylabel("Residue")
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Contact map saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot contact map: {str(e)}")
            raise
    
    def plot_secondary_structure(
        self,
        structure_file: str,
        output_file: str,
        title: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """Plot secondary structure assignment."""
        try:
            logger.debug("Plotting secondary structure")
            
            # Parse structure
            parser = PDBParser()
            structure = parser.get_structure('protein', structure_file)
            
            # Extract secondary structure
            model = structure[0]
            ss = []
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                        ss.append(residue.get_ss())
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Plot secondary structure
            ss_map = {'H': 1, 'B': 0.5, 'E': 0, 'G': 0.75, 'I': 0.25, 'T': 0.125, 'S': 0.125, '-': 0}
            ss_values = [ss_map[s] for s in ss]
            
            ax.plot(range(len(ss_values)), ss_values, 'k-', linewidth=2)
            ax.fill_between(range(len(ss_values)), ss_values, alpha=0.3)
            
            # Add secondary structure labels
            ax.set_yticks([0, 0.125, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(['E', 'T/S', 'I', 'B', 'G', 'H'])
            
            # Customize plot
            plot_title = title or "Secondary Structure"
            if model_name:
                plot_title = self._add_version_info_to_title(plot_title, model_name)
            ax.set_title(plot_title)
            ax.set_xlabel("Residue Position")
            ax.set_ylabel("Secondary Structure")
            ax.grid(True, alpha=0.3)
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Secondary structure plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot secondary structure: {str(e)}")
            raise
    
    def plot_structure_comparison(
        self,
        structures: Dict[str, str],
        output_file: str,
        reference_structure: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """Plot comprehensive structure comparison metrics."""
        try:
            if not self.config.use_comparison:
                logger.info("Skipping structure comparison visualization (UseComparison=False)")
                return
                
            logger.debug("Plotting structure comparison metrics")
            
            # Use first structure as reference if none provided
            if not reference_structure:
                reference_structure = list(structures.values())[0]
            
            # Calculate comparison metrics
            metrics = {}
            for model_name, structure_file in structures.items():
                if structure_file != reference_structure:
                    metrics[model_name] = self._get_structure_metrics(structure_file, reference_structure)
            
            # Create figure with subplots
            fig, axes = self._create_figure(2, 2)
            axes = axes.ravel()
            
            # Plot metrics
            metrics_to_plot = {
                'RMSD': 'rmsd',
                'TM-Score': 'tm_score',
                'Contact Overlap': 'contact_overlap',
                'SS Similarity': 'ss_similarity'
            }
            
            for i, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
                values = [m[metric_key] for m in metrics.values()]
                model_names = list(metrics.keys())
                
                axes[i].bar(model_names, values)
                axes[i].set_title(metric_name)
                axes[i].set_xlabel("Model")
                axes[i].set_ylabel("Score")
                axes[i].grid(True, alpha=0.3)
                
                # Rotate x-axis labels for better readability
                axes[i].tick_params(axis='x', rotation=45)
            
            # Add overall title
            plot_title = title or "Structure Comparison Metrics"
            fig.suptitle(plot_title)
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Structure comparison plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot structure comparison: {str(e)}")
            raise 