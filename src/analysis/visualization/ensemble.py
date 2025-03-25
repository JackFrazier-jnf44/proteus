"""Ensemble visualization module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseVisualizer, VisualizationConfig
from ..ensemble import EnsemblePredictor, EnsembleConfig

logger = logging.getLogger(__name__)

class EnsembleVisualizer(BaseVisualizer):
    """Specialized visualizer for ensemble analysis."""
    
    def plot_ensemble_predictions(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        ensemble_config: EnsembleConfig,
        output_file: str,
        title: Optional[str] = None
    ) -> None:
        """Plot ensemble predictions and individual model predictions."""
        try:
            if not self.config.use_ensemble:
                logger.info("Skipping ensemble visualization (UseEnsemble=False)")
                return
                
            logger.debug("Plotting ensemble predictions")
            
            # Create ensemble predictor
            ensemble = EnsemblePredictor(ensemble_config)
            
            # Get combined predictions
            combined = ensemble.combine_predictions(predictions)
            
            # Create figure with subplots
            fig, (ax1, ax2) = self._create_figure(2, 1)
            
            # Plot individual model predictions
            for model_name, pred in predictions.items():
                ax1.plot(pred['structure'], label=model_name, alpha=0.7)
            
            # Plot ensemble prediction
            ax1.plot(combined['structure'], label='Ensemble', linewidth=2, color='black')
            
            # Customize first subplot
            plot_title = title or "Model and Ensemble Predictions"
            ax1.set_title(plot_title)
            ax1.set_xlabel("Residue Position")
            ax1.set_ylabel("Structure")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot confidence scores
            if 'confidence' in predictions[list(predictions.keys())[0]]:
                for model_name, pred in predictions.items():
                    ax2.plot(pred['confidence'], label=model_name, alpha=0.7)
                
                if 'confidence' in combined:
                    ax2.plot(combined['confidence'], label='Ensemble', linewidth=2, color='black')
                
                ax2.set_title("Confidence Scores")
                ax2.set_xlabel("Residue Position")
                ax2.set_ylabel("Confidence")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_file)
            
            logger.debug(f"Ensemble predictions plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot ensemble predictions: {str(e)}")
            raise
    
    def plot_ensemble_comparison(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        ensemble_config: EnsembleConfig,
        structures: Dict[str, str],
        output_file: str,
        reference_structure: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """Plot ensemble predictions and structure comparison metrics together."""
        try:
            if not (self.config.use_ensemble and self.config.use_comparison):
                logger.info("Skipping ensemble comparison visualization (UseEnsemble=False or UseComparison=False)")
                return
                
            logger.debug("Plotting ensemble comparison")
            
            # Create figure with subplots
            fig = plt.figure(figsize=self.config.figure_size)
            gs = fig.add_gridspec(3, 2)
            
            # Plot ensemble predictions
            ax1 = fig.add_subplot(gs[0, :])
            self.plot_ensemble_predictions(predictions, ensemble_config, None, title)
            
            # Plot structure comparison
            ax2 = fig.add_subplot(gs[1:, :])
            self.plot_structure_comparison(structures, None, reference_structure, None)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_file)
            
            logger.debug(f"Ensemble comparison plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot ensemble comparison: {str(e)}")
            raise
    
    def plot_ensemble_weights(
        self,
        ensemble_config: EnsembleConfig,
        output_file: str,
        title: Optional[str] = None
    ) -> None:
        """Plot ensemble weights distribution."""
        try:
            if not self.config.use_ensemble:
                logger.info("Skipping ensemble weights visualization (UseEnsemble=False)")
                return
                
            logger.debug("Plotting ensemble weights")
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Get weights
            weights = ensemble_config.weights or {
                model: 1.0/len(ensemble_config.models)
                for model in ensemble_config.models
            }
            
            # Plot weights
            model_names = list(weights.keys())
            weight_values = list(weights.values())
            
            ax.bar(model_names, weight_values)
            
            # Customize plot
            plot_title = title or "Ensemble Weights"
            ax.set_title(plot_title)
            ax.set_xlabel("Model")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Ensemble weights plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot ensemble weights: {str(e)}")
            raise 