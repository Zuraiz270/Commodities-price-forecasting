"""
SHAP Analysis Module

Model interpretability using SHapley Additive exPlanations.
Provides feature importance analysis for time series forecasting models.

Best practices for time series SHAP (2024-2025):
- Handle temporal dependencies carefully
- Use appropriate SHAP explainers for model types
- Consider time-series-specific visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union, Callable
import warnings


class SHAPAnalyzer:
    """
    SHAP-based model interpretability for commodity forecasting.
    
    Supports:
    - Tree-based models (XGBoost, Random Forest, LightGBM)
    - Deep learning models (LSTM, TFT, N-BEATS)
    - Model-agnostic explanations
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str = "auto",
        feature_names: List[str] = None
    ):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model
            model_type: 'tree', 'deep', 'kernel', or 'auto'
            feature_names: Names of input features
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type."""
        model_class = type(self.model).__name__.lower()
        
        if any(name in model_class for name in ['xgb', 'lgb', 'catboost', 'forest', 'tree']):
            return 'tree'
        elif any(name in model_class for name in ['lstm', 'tft', 'nbeats', 'nhits', 'neural']):
            return 'deep'
        else:
            return 'kernel'
    
    def fit(
        self,
        background_data: np.ndarray,
        max_samples: int = 100
    ) -> 'SHAPAnalyzer':
        """
        Fit the SHAP explainer with background data.
        
        Args:
            background_data: Background dataset for SHAP
            max_samples: Max samples for background (for efficiency)
            
        Returns:
            Self for chaining
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        if self.model_type == "auto":
            self.model_type = self._detect_model_type()
        
        # Subsample background if needed
        if len(background_data) > max_samples:
            indices = np.random.choice(len(background_data), max_samples, replace=False)
            background_data = background_data[indices]
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data)
        else:
            # Kernel SHAP for model-agnostic explanations
            self.explainer = shap.KernelExplainer(
                self._model_predict,
                shap.sample(background_data, min(max_samples, 50))
            )
        
        return self
    
    def _model_predict(self, x: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction."""
        import torch
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            # PyTorch model
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                output = self.model(x_tensor)
                if isinstance(output, dict):
                    pred = output.get('predictions', output.get('forecast'))
                else:
                    pred = output
                return pred.numpy()
    
    def explain(
        self,
        X: np.ndarray,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Must call fit() before explain()")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.shap_values = self.explainer.shap_values(
                X,
                check_additivity=check_additivity
            )
        
        return self.shap_values
    
    def feature_importance(
        self,
        X: np.ndarray = None,
        importance_type: str = "mean_abs"
    ) -> pd.DataFrame:
        """
        Compute feature importance from SHAP values.
        
        Args:
            X: Data to compute importance for (uses stored if None)
            importance_type: 'mean_abs' or 'mean'
            
        Returns:
            DataFrame with feature importance scores
        """
        if X is not None:
            self.explain(X)
        
        if self.shap_values is None:
            raise ValueError("Must call explain() first or provide X")
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        # Compute importance
        if importance_type == "mean_abs":
            importance = np.abs(shap_vals).mean(axis=0)
        else:
            importance = shap_vals.mean(axis=0)
        
        # Handle multi-dimensional SHAP values (time series)
        if importance.ndim > 1:
            importance = importance.mean(axis=tuple(range(1, importance.ndim)))
        
        # Create DataFrame
        if self.feature_names:
            names = self.feature_names[:len(importance)]
        else:
            names = [f"Feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            "feature": names,
            "importance": importance
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    
    def plot_importance(
        self,
        X: np.ndarray = None,
        top_k: int = 20,
        figsize: tuple = (10, 8),
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            X: Data to compute importance for
            top_k: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        importance_df = self.feature_importance(X)
        top_features = importance_df.head(top_k)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(
            range(len(top_features)),
            top_features["importance"],
            color='steelblue'
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (SHAP)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_summary(
        self,
        X: np.ndarray,
        plot_type: str = "dot",
        max_display: int = 20,
        save_path: str = None
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to explain
            plot_type: 'dot', 'bar', or 'violin'
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        shap_vals = self.explain(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        # Flatten if needed for time series
        if shap_vals.ndim > 2:
            original_shape = shap_vals.shape
            shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_vals,
            X_flat,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def temporal_importance(
        self,
        X: np.ndarray,
        time_index: pd.DatetimeIndex = None
    ) -> pd.DataFrame:
        """
        Analyze feature importance over time.
        
        Useful for understanding how feature contributions
        change across different time periods.
        
        Args:
            X: Time series data [samples, timesteps, features]
            time_index: Optional datetime index
            
        Returns:
            DataFrame with temporal importance patterns
        """
        shap_vals = self.explain(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        if shap_vals.ndim < 3:
            print("Warning: Input doesn't appear to be 3D time series data")
            return self.feature_importance(X)
        
        n_samples, n_timesteps, n_features = shap_vals.shape
        
        # Aggregate importance per timestep
        temporal_importance = np.abs(shap_vals).mean(axis=0).mean(axis=-1)
        
        if time_index is None:
            time_index = list(range(n_timesteps))
        
        return pd.DataFrame({
            "timestep": time_index[-n_timesteps:],
            "importance": temporal_importance
        })
    
    def get_explanation_for_sample(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a single sample.
        
        Args:
            X: Data containing the sample
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        shap_vals = self.explain(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        
        sample_shap = shap_vals[sample_idx]
        sample_features = X[sample_idx]
        
        # Create feature contribution summary
        if sample_shap.ndim > 1:
            # Aggregate across time for summary
            contributions = np.abs(sample_shap).mean(axis=0)
        else:
            contributions = sample_shap
        
        if self.feature_names:
            names = self.feature_names[:len(contributions)]
        else:
            names = [f"Feature_{i}" for i in range(len(contributions))]
        
        return {
            "shap_values": sample_shap,
            "feature_values": sample_features,
            "feature_contributions": dict(zip(names, contributions)),
            "top_positive": sorted(
                zip(names, contributions),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_negative": sorted(
                zip(names, contributions),
                key=lambda x: x[1]
            )[:5]
        }
