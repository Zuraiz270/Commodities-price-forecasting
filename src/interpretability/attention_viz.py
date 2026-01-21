"""
Attention Visualization Module

Visualize attention weights from Transformer-based models (TFT, LSTM-Attention)
for interpretable time series forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Any, Tuple
import torch


class AttentionVisualizer:
    """
    Visualize attention patterns from Transformer-based forecasting models.
    
    Supports:
    - TFT attention weights
    - LSTM-Attention attention maps
    - Variable selection network outputs
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        x_labels: List[str] = None,
        y_labels: List[str] = None,
        title: str = "Attention Weights",
        cmap: str = "Blues",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention matrix [query_len, key_len]
            x_labels: Labels for keys (horizontal axis)
            y_labels: Labels for queries (vertical axis)
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(attention_weights, cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
        
        # Set labels
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        
        ax.set_xlabel("Key Position (Past)")
        ax.set_ylabel("Query Position (Future)")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_attention(
        self,
        attention_weights: np.ndarray,
        time_index: pd.DatetimeIndex = None,
        highlight_periods: List[Tuple[int, int]] = None,
        title: str = "Temporal Attention Pattern",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot how attention is distributed across time.
        
        Args:
            attention_weights: Attention weights [batch, query_len, key_len] or [query_len, key_len]
            time_index: Optional datetime index for x-axis
            highlight_periods: List of (start, end) tuples to highlight
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Average across batch if needed
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)
        
        # Average across query positions
        temporal_attention = attention_weights.mean(axis=0)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = time_index if time_index is not None else range(len(temporal_attention))
        
        ax.fill_between(x, temporal_attention, alpha=0.3, color='steelblue')
        ax.plot(x, temporal_attention, color='steelblue', linewidth=2)
        
        # Highlight periods if specified
        if highlight_periods:
            for start, end in highlight_periods:
                ax.axvspan(start, end, alpha=0.2, color='red')
        
        ax.set_xlabel("Time" if time_index is None else "Date")
        ax.set_ylabel("Average Attention Weight")
        ax.set_title(title)
        
        if time_index is not None:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_variable_importance(
        self,
        variable_weights: np.ndarray,
        feature_names: List[str] = None,
        top_k: int = 15,
        title: str = "Variable Selection (TFT)",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot TFT variable selection network importance.
        
        Args:
            variable_weights: Variable selection weights [batch, time, features]
            feature_names: Names of features
            top_k: Number of top features to display
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Average across batch and time
        if variable_weights.ndim > 1:
            importance = variable_weights.mean(axis=tuple(range(variable_weights.ndim - 1)))
        else:
            importance = variable_weights
        
        n_features = len(importance)
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        top_idx = sorted_idx[:top_k]
        
        top_names = [feature_names[i] for i in top_idx]
        top_importance = importance[top_idx]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(range(len(top_names)), top_importance, color='teal')
        
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel("Selection Weight")
        ax.set_title(title)
        
        # Add value labels
        for bar, val in zip(bars, top_importance):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_attention_over_horizons(
        self,
        attention_weights: np.ndarray,
        horizons: List[int] = None,
        title: str = "Attention by Forecast Horizon",
        save_path: str = None
    ) -> plt.Figure:
        """
        Compare attention patterns across different forecast horizons.
        
        Args:
            attention_weights: Attention weights [horizons, input_length] or [batch, horizons, input_length]
            horizons: Horizon labels
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Average across batch if needed
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)
        
        n_horizons, input_length = attention_weights.shape
        
        if horizons is None:
            horizons = [f"t+{i+1}" for i in range(n_horizons)]
        
        fig, axes = plt.subplots(
            nrows=min(n_horizons, 6),
            ncols=1,
            figsize=(12, 2 * min(n_horizons, 6)),
            sharex=True
        )
        
        if n_horizons == 1:
            axes = [axes]
        
        for i, (ax, horizon) in enumerate(zip(axes, horizons)):
            ax.fill_between(
                range(input_length),
                attention_weights[i],
                alpha=0.3,
                color=plt.cm.viridis(i / n_horizons)
            )
            ax.plot(
                attention_weights[i],
                color=plt.cm.viridis(i / n_horizons),
                linewidth=1.5
            )
            ax.set_ylabel(f"{horizon}")
            ax.set_ylim(0, None)
        
        axes[-1].set_xlabel("Input Timestep (t-N to t)")
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_multi_head_attention(
        self,
        attention_weights: np.ndarray,
        head_names: List[str] = None,
        title: str = "Multi-Head Attention",
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize attention from multiple heads.
        
        Args:
            attention_weights: Attention weights [num_heads, query_len, key_len]
            head_names: Names for each head
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_heads = attention_weights.shape[0]
        
        if head_names is None:
            head_names = [f"Head {i+1}" for i in range(num_heads)]
        
        # Determine grid layout
        ncols = min(4, num_heads)
        nrows = (num_heads + ncols - 1) // ncols
        
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 3 * nrows)
        )
        
        if num_heads == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        
        for idx in range(num_heads):
            row, col = idx // ncols, idx % ncols
            ax = axes[row][col] if nrows > 1 else axes[col]
            
            im = ax.imshow(attention_weights[idx], cmap='Blues', aspect='auto')
            ax.set_title(head_names[idx])
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
        
        # Hide unused subplots
        for idx in range(num_heads, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            ax = axes[row][col] if nrows > 1 else axes[col]
            ax.axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def extract_attention_from_model(
        self,
        model: torch.nn.Module,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from a model.
        
        Args:
            model: PyTorch model with attention
            x: Input tensor
            
        Returns:
            Dictionary with extracted attention weights
        """
        model.eval()
        
        with torch.no_grad():
            output = model(x, return_attention=True)
        
        results = {}
        
        if isinstance(output, dict):
            if "attention_weights" in output:
                results["attention"] = output["attention_weights"].cpu().numpy()
            
            if "variable_weights" in output:
                results["variable_selection"] = output["variable_weights"].cpu().numpy()
        
        return results
    
    def create_attention_report(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        feature_names: List[str] = None,
        save_dir: str = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive attention visualization report.
        
        Args:
            model: Model with attention
            x: Input data
            feature_names: Feature names
            save_dir: Directory to save figures
            
        Returns:
            Dictionary of generated figures
        """
        attention_data = self.extract_attention_from_model(model, x)
        figures = {}
        
        if "attention" in attention_data:
            attn = attention_data["attention"]
            
            # Average attention heatmap
            figures["heatmap"] = self.plot_attention_heatmap(
                attn.mean(axis=0),
                title="Average Attention Heatmap",
                save_path=f"{save_dir}/attention_heatmap.png" if save_dir else None
            )
            
            # Temporal attention
            figures["temporal"] = self.plot_temporal_attention(
                attn,
                title="Temporal Attention Distribution",
                save_path=f"{save_dir}/temporal_attention.png" if save_dir else None
            )
        
        if "variable_selection" in attention_data:
            figures["variables"] = self.plot_variable_importance(
                attention_data["variable_selection"],
                feature_names=feature_names,
                title="Variable Selection Importance",
                save_path=f"{save_dir}/variable_importance.png" if save_dir else None
            )
        
        return figures
