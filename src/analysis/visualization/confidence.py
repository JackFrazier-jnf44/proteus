"""Confidence visualization module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseVisualizer, VisualizationConfig

logger = logging.getLogger(__name__)

class ConfidenceVisualizer(BaseVisualizer):
    """Specialized visualizer for confidence analysis."""
    
    def plot_confidence_per_residue(
        self,
        confidence_scores: Dict[str, np.ndarray],
        sequence: str,
        output_file: str,
        title: Optional[str] = None
    ) -> None:
        """Plot confidence scores per residue for different models."""
        try:
            if not self.config.call_model:
                logger.info("Skipping confidence score visualization (CallModel=False)")
                return
                
            logger.debug("Plotting confidence scores per residue")
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Plot confidence scores for each model
            for model_name, scores in confidence_scores.items():
                ax.plot(range(len(sequence)), scores, label=model_name, alpha=0.7)
            
            # Customize plot
            plot_title = title or "Confidence Scores per Residue"
            ax.set_title(plot_title)
            ax.set_xlabel("Residue Position")
            ax.set_ylabel("Confidence Score")
            ax.set_xticks(range(len(sequence)))
            ax.set_xticklabels(sequence, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Confidence scores plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot confidence scores: {str(e)}")
            raise
    
    def plot_confidence_distribution(
        self,
        confidence_scores: Dict[str, np.ndarray],
        output_file: str,
        title: Optional[str] = None
    ) -> None:
        """Plot distribution of confidence scores across models."""
        try:
            if not self.config.call_model:
                logger.info("Skipping confidence distribution visualization (CallModel=False)")
                return
                
            logger.debug("Plotting confidence score distribution")
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Plot confidence distributions
            for model_name, scores in confidence_scores.items():
                sns.kdeplot(scores, label=model_name, ax=ax)
            
            # Customize plot
            plot_title = title or "Confidence Score Distribution"
            ax.set_title(plot_title)
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Confidence distribution plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot confidence distribution: {str(e)}")
            raise
    
    def plot_confidence_comparison(
        self,
        confidence_scores: Dict[str, np.ndarray],
        output_file: str,
        title: Optional[str] = None
    ) -> None:
        """Plot comprehensive confidence analysis."""
        try:
            if not self.config.call_model:
                logger.info("Skipping confidence comparison visualization (CallModel=False)")
                return
                
            logger.debug("Plotting confidence comparison")
            
            # Create figure with subplots
            fig, (ax1, ax2) = self._create_figure(2, 1)
            
            # Plot confidence per residue
            self.plot_confidence_per_residue(confidence_scores, None, None, title)
            
            # Plot confidence distribution
            self.plot_confidence_distribution(confidence_scores, None, None)
            
            # Calculate and plot summary statistics
            stats = {}
            for model_name, scores in confidence_scores.items():
                stats[model_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            # Plot summary statistics
            model_names = list(stats.keys())
            means = [s['mean'] for s in stats.values()]
            stds = [s['std'] for s in stats.values()]
            
            ax2.bar(model_names, means, yerr=stds, capsize=5)
            ax2.set_title("Confidence Score Statistics")
            ax2.set_xlabel("Model")
            ax2.set_ylabel("Mean Confidence Score")
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax2.tick_params(axis='x', rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_file)
            
            logger.debug(f"Confidence comparison plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot confidence comparison: {str(e)}")
            raise 