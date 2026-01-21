"""
Visualization Utilities

Plotting functions for model results and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path


class Visualizer:
    """
    Visualization utilities for commodity price forecasting.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        style: str = "seaborn-v0_8-whitegrid"
    ):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex = None,
        title: str = "Prediction vs Actual",
        confidence_interval: Tuple[np.ndarray, np.ndarray] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot predictions against actual values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Date index for x-axis
            title: Plot title
            confidence_interval: Tuple of (lower, upper) bounds
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = dates if dates is not None else range(len(y_true))
        
        ax.plot(x, y_true, label='Actual', color='steelblue', linewidth=2)
        ax.plot(x, y_pred, label='Predicted', color='coral', linewidth=2, linestyle='--')
        
        if confidence_interval:
            lower, upper = confidence_interval
            ax.fill_between(x, lower, upper, alpha=0.2, color='coral', label='95% CI')
        
        ax.set_xlabel('Date' if dates is not None else 'Time')
        ax.set_ylabel('Price')
        ax.set_title(title)
        ax.legend()
        
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot prediction residuals.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Residuals over time
        axes[0].plot(residuals, color='steelblue', alpha=0.7)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_title('Residuals Over Time')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Residual')
        
        # Histogram
        axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_title('Residual Distribution')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(
        self,
        results: pd.DataFrame,
        metric: str = "RMSE",
        save_path: str = None
    ) -> plt.Figure:
        """
        Compare model performance.
        
        Args:
            results: DataFrame with model names and metrics
            metric: Metric to compare
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))
        
        bars = ax.barh(results['model'], results[metric], color=colors)
        
        ax.set_xlabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        
        # Add value labels
        for bar, val in zip(bars, results[metric]):
            ax.text(
                bar.get_width() + 0.01 * max(results[metric]),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}',
                va='center'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: Dictionary with 'train_loss' and optionally 'val_loss'
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
        
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], label='Validation Loss', color='coral')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_forecast_horizons(
        self,
        predictions: Dict[int, np.ndarray],
        actuals: Dict[int, np.ndarray],
        dates: pd.DatetimeIndex = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot predictions for different forecast horizons.
        
        Args:
            predictions: Dict mapping horizon to predictions
            actuals: Dict mapping horizon to actual values
            dates: Date index
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_horizons = len(predictions)
        ncols = min(3, n_horizons)
        nrows = (n_horizons + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        
        if n_horizons == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        
        for idx, (horizon, pred) in enumerate(predictions.items()):
            row, col = idx // ncols, idx % ncols
            ax = axes[row][col] if nrows > 1 else axes[col]
            
            actual = actuals.get(horizon, None)
            x = dates if dates is not None else range(len(pred))
            
            if actual is not None:
                ax.plot(x, actual, label='Actual', alpha=0.7)
            ax.plot(x, pred, label='Predicted', linestyle='--')
            
            ax.set_title(f'Lead Time: {horizon}')
            ax.legend()
        
        # Hide unused
        for idx in range(n_horizons, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            ax = axes[row][col] if nrows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_commodity_prices(
        self,
        data: pd.DataFrame,
        commodities: List[str] = None,
        normalize: bool = False,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot multiple commodity prices.
        
        Args:
            data: DataFrame with commodity prices
            commodities: List of commodities to plot
            normalize: Whether to normalize for comparison
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        commodities = commodities or list(data.columns)
        
        for commodity in commodities:
            if commodity in data.columns:
                values = data[commodity]
                if normalize:
                    values = (values - values.min()) / (values.max() - values.min())
                ax.plot(data.index, values, label=commodity, linewidth=1.5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price' + (' (Normalized)' if normalize else ''))
        ax.set_title('Commodity Prices')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
