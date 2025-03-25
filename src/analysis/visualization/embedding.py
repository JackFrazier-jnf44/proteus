"""Embedding visualization module."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseVisualizer, VisualizationConfig

logger = logging.getLogger(__name__)

class EmbeddingVisualizer(BaseVisualizer):
    """Specialized visualizer for embedding analysis."""
    
    def plot_embedding_space(
        self,
        embeddings: Dict[str, np.ndarray],
        output_file: str,
        title: Optional[str] = None,
        n_components: int = 2
    ) -> None:
        """Plot embedding space visualization using t-SNE."""
        try:
            if not self.config.call_model:
                logger.info("Skipping embedding space visualization (CallModel=False)")
                return
                
            logger.debug("Plotting embedding space")
            
            # Combine embeddings from all models
            all_embeddings = []
            model_labels = []
            
            for model_name, embedding in embeddings.items():
                # Standardize embeddings
                embedding_scaled = self._standardize_embeddings(embedding)
                
                all_embeddings.append(embedding_scaled)
                model_labels.extend([model_name] * len(embedding))
            
            # Combine all embeddings
            X = np.vstack(all_embeddings)
            
            # Apply t-SNE
            X_tsne = self._apply_tsne(X, n_components)
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Plot embeddings
            for label in set(model_labels):
                mask = [l == label for l in model_labels]
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.6)
            
            # Customize plot
            plot_title = title or "Embedding Space (t-SNE)"
            ax.set_title(plot_title)
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.legend()
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Embedding space plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot embedding space: {str(e)}")
            raise
    
    def plot_embedding_comparison(
        self,
        embeddings: Dict[str, np.ndarray],
        output_file: str,
        title: Optional[str] = None
    ) -> None:
        """Plot comparison of embeddings from different models."""
        try:
            if not self.config.call_model:
                logger.info("Skipping embedding comparison visualization (CallModel=False)")
                return
                
            logger.debug("Plotting embedding comparison")
            
            # Create figure with subplots
            fig, (ax1, ax2) = self._create_figure(1, 2)
            
            # Plot t-SNE visualization
            self.plot_embedding_space(embeddings, None, title)
            
            # Calculate and plot embedding distances
            distances = {}
            for model_name, embedding in embeddings.items():
                embedding_scaled = self._standardize_embeddings(embedding)
                distances[model_name] = np.mean(np.linalg.norm(embedding_scaled, axis=1))
            
            # Plot distances
            model_names = list(distances.keys())
            distance_values = list(distances.values())
            
            ax2.bar(model_names, distance_values)
            ax2.set_title("Average Embedding Norm")
            ax2.set_xlabel("Model")
            ax2.set_ylabel("Average Norm")
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax2.tick_params(axis='x', rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_file)
            
            logger.debug(f"Embedding comparison plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot embedding comparison: {str(e)}")
            raise
    
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        sequence: str,
        output_file: str,
        title: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """Plot attention weights from transformer models."""
        try:
            if not self.config.call_model:
                logger.info("Skipping attention weight visualization (CallModel=False)")
                return
                
            logger.debug("Plotting attention weights")
            
            # Create figure
            fig, ax = self._create_figure()
            
            # Plot attention weights
            sns.heatmap(
                attention_weights,
                cmap='viridis',
                xticklabels=sequence,
                yticklabels=sequence,
                ax=ax
            )
            
            # Customize plot
            plot_title = title or "Attention Weights"
            if model_name:
                plot_title = self._add_version_info_to_title(plot_title, model_name)
            ax.set_title(plot_title)
            ax.set_xlabel("Target Residue")
            ax.set_ylabel("Source Residue")
            
            # Save plot
            self._save_figure(fig, output_file)
            
            logger.debug(f"Attention weights plot saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot attention weights: {str(e)}")
            raise 