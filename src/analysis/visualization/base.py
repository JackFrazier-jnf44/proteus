"""Base visualization module with core functionality."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import seaborn as sns
from dataclasses import dataclass, field
from Bio.PDB import *
from MDAnalysis import Universe
from MDAnalysis.analysis import rms
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import json
import csv
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime

from ...config import PLOT_SETTINGS, ANALYSIS_SETTINGS
from ..ensemble import EnsemblePredictor, EnsembleConfig
from ..structure.comparison import StructureComparer
from ..versioning import ModelVersionManager
from ..memory import MemoryManager

logger = logging.getLogger(__name__)

@dataclass
class ExportConfig:
    """Configuration for export settings."""
    format: str = 'png'  # Export format (png, jpg, pdf, svg, html, json, csv)
    quality: int = 95  # Quality for lossy formats (jpg)
    compression: int = 6  # Compression level for PNG
    metadata: bool = True  # Include metadata in exports
    interactive: bool = False  # Enable interactive features
    web_gl: bool = False  # Use WebGL for 3D visualizations
    responsive: bool = True  # Make HTML exports responsive

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: Tuple[float, float] = (15, 10)
    dpi: int = 300
    style: str = 'seaborn'
    color_palette: str = 'Set2'
    output_format: str = 'png'
    call_model: bool = True
    use_ensemble: bool = True
    use_comparison: bool = True
    memory_limit: Optional[int] = None
    export_config: ExportConfig = field(default_factory=ExportConfig)

class BaseVisualizer:
    """Base class for all visualizations with integrated memory management and versioning."""
    
    def __init__(
        self,
        config: VisualizationConfig,
        version_manager: Optional[ModelVersionManager] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Initialize base visualizer.
        
        Args:
            config: Visualization configuration
            version_manager: Optional model version manager
            memory_manager: Optional memory manager
        """
        self.config = config
        self.version_manager = version_manager
        self.memory_manager = memory_manager or MemoryManager(
            memory_limit=config.memory_limit
        )
        self.structure_comparer = StructureComparer()
        
        # Set up matplotlib style
        plt.style.use(config.style)
        sns.set_palette(config.color_palette)
        
        # Initialize export formats
        self._supported_formats = {
            'png': self._export_png,
            'jpg': self._export_jpg,
            'pdf': self._export_pdf,
            'svg': self._export_svg,
            'html': self._export_html,
            'json': self._export_json,
            'csv': self._export_csv
        }
    
    def _create_figure(self, nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, np.ndarray]:
        """Create a figure with specified number of subplots."""
        return plt.subplots(nrows, ncols, figsize=self.config.figure_size)
    
    def _save_figure(self, fig: plt.Figure, output_file: str) -> None:
        """Save figure with memory management and multiple format support."""
        try:
            with self.memory_manager.monitor():
                # Get file extension and format
                output_path = Path(output_file)
                format_ext = output_path.suffix[1:].lower()
                
                # Check if format is supported
                if format_ext not in self._supported_formats:
                    raise ValueError(f"Unsupported export format: {format_ext}")
                
                # Export using appropriate method
                self._supported_formats[format_ext](fig, output_path)
                
                # Close figure
                plt.close(fig)
                
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
            raise
    
    def _export_png(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure as PNG."""
        fig.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            format='png',
            optimize=True,
            compression=self.config.export_config.compression
        )
    
    def _export_jpg(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure as JPG."""
        fig.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            format='jpg',
            quality=self.config.export_config.quality
        )
    
    def _export_pdf(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure as PDF."""
        fig.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            format='pdf'
        )
    
    def _export_svg(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure as SVG."""
        fig.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            format='svg'
        )
    
    def _export_html(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure as interactive HTML."""
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            
            # Convert matplotlib figure to plotly
            fig_data = self._convert_to_plotly(fig)
            
            # Create HTML with metadata if requested
            html_content = pio.to_html(
                fig_data,
                full_html=True,
                include_plotlyjs=True,
                config={
                    'responsive': self.config.export_config.responsive,
                    'displayModeBar': self.config.export_config.interactive,
                    'scrollZoom': self.config.export_config.interactive
                }
            )
            
            # Add metadata if requested
            if self.config.export_config.metadata:
                metadata = self._get_plot_metadata(fig)
                html_content = self._add_metadata_to_html(html_content, metadata)
            
            # Save HTML file
            output_path.write_text(html_content)
            
        except ImportError:
            logger.warning("Plotly not installed. Falling back to static HTML export.")
            self._export_static_html(fig, output_path)
    
    def _export_static_html(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure as static HTML."""
        # Convert figure to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.config.dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Protein Structure Visualization</title>
            <style>
                body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                img {{ max-width: 100%; height: auto; }}
                .metadata {{ margin-top: 20px; padding: 10px; background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <img src="data:image/png;base64,{img_base64}" alt="Protein Structure Visualization">
                {self._get_metadata_html(fig) if self.config.export_config.metadata else ''}
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html_content)
    
    def _export_json(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure data as JSON."""
        # Extract data from figure
        data = self._extract_figure_data(fig)
        
        # Add metadata if requested
        if self.config.export_config.metadata:
            data['metadata'] = self._get_plot_metadata(fig)
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, fig: plt.Figure, output_path: Path) -> None:
        """Export figure data as CSV."""
        # Extract data from figure
        data = self._extract_figure_data(fig)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save CSV
        df.to_csv(output_path, index=False)
    
    def _convert_to_plotly(self, fig: plt.Figure) -> go.Figure:
        """Convert matplotlib figure to plotly figure."""
        import plotly.graph_objects as go
        
        # Extract data from matplotlib figure
        data = []
        for ax in fig.axes:
            for line in ax.lines:
                data.append(go.Scatter(
                    x=line.get_xdata(),
                    y=line.get_ydata(),
                    mode='lines',
                    name=line.get_label()
                ))
            
            for collection in ax.collections:
                if isinstance(collection, plt.matplotlib.collections.PathCollection):
                    data.append(go.Scatter(
                        x=collection.get_offsets()[:, 0],
                        y=collection.get_offsets()[:, 1],
                        mode='markers',
                        name=collection.get_label()
                    ))
        
        # Create plotly figure
        return go.Figure(data=data)
    
    def _extract_figure_data(self, fig: plt.Figure) -> Dict[str, Any]:
        """Extract data from matplotlib figure."""
        data = {}
        
        for i, ax in enumerate(fig.axes):
            ax_data = {}
            
            # Extract lines
            for j, line in enumerate(ax.lines):
                ax_data[f'line_{j}'] = {
                    'x': line.get_xdata().tolist(),
                    'y': line.get_ydata().tolist(),
                    'label': line.get_label()
                }
            
            # Extract scatter points
            for j, collection in enumerate(ax.collections):
                if isinstance(collection, plt.matplotlib.collections.PathCollection):
                    ax_data[f'scatter_{j}'] = {
                        'x': collection.get_offsets()[:, 0].tolist(),
                        'y': collection.get_offsets()[:, 1].tolist(),
                        'label': collection.get_label()
                    }
            
            data[f'axis_{i}'] = ax_data
        
        return data
    
    def _get_plot_metadata(self, fig: plt.Figure) -> Dict[str, Any]:
        """Get metadata for the plot."""
        return {
            'title': fig.suptitle.get_text() if fig.suptitle else None,
            'axes': [ax.get_title() for ax in fig.axes],
            'figure_size': fig.get_size_inches().tolist(),
            'dpi': self.config.dpi,
            'style': self.config.style,
            'color_palette': self.config.color_palette,
            'timestamp': datetime.now().isoformat()
        }
    
    def _add_metadata_to_html(self, html_content: str, metadata: Dict[str, Any]) -> str:
        """Add metadata to HTML content."""
        metadata_html = f"""
        <div class="metadata">
            <h3>Plot Metadata</h3>
            <pre>{json.dumps(metadata, indent=2)}</pre>
        </div>
        """
        return html_content.replace('</body>', f'{metadata_html}</body>')
    
    def _get_metadata_html(self, fig: plt.Figure) -> str:
        """Get metadata HTML for static HTML export."""
        metadata = self._get_plot_metadata(fig)
        return f"""
        <div class="metadata">
            <h3>Plot Metadata</h3>
            <pre>{json.dumps(metadata, indent=2)}</pre>
        </div>
        """
    
    def _get_model_version_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get version information for a model if version manager is available."""
        if self.version_manager:
            try:
                version = self.version_manager.get_latest_version(model_name)
                return {
                    'version_id': version.version_id,
                    'created_at': version.created_at,
                    'description': version.description,
                    'tags': version.tags
                }
            except Exception as e:
                logger.warning(f"Failed to get version info for {model_name}: {str(e)}")
        return None
    
    def _add_version_info_to_title(self, title: str, model_name: str) -> str:
        """Add version information to plot title if available."""
        version_info = self._get_model_version_info(model_name)
        if version_info:
            return f"{title}\nVersion: {version_info['version_id']} ({version_info['created_at']})"
        return title
    
    def _standardize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Standardize embeddings for visualization."""
        if len(embeddings.shape) > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        scaler = StandardScaler()
        return scaler.fit_transform(embeddings)
    
    def _apply_tsne(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Apply t-SNE to embeddings."""
        tsne = TSNE(n_components=n_components, random_state=42)
        return tsne.fit_transform(embeddings)
    
    def _get_structure_metrics(
        self,
        structure_file: str,
        reference_structure: Optional[str] = None
    ) -> Dict[str, float]:
        """Get comprehensive structure metrics."""
        metrics = {}
        if reference_structure:
            metrics.update({
                'rmsd': self.structure_comparer.calculate_rmsd(reference_structure, structure_file),
                'tm_score': self.structure_comparer.calculate_tm_score(reference_structure, structure_file),
                'contact_overlap': self.structure_comparer.calculate_contact_map_overlap(reference_structure, structure_file),
                'ss_similarity': self.structure_comparer.calculate_secondary_structure_similarity(reference_structure, structure_file)
            })
        return metrics 