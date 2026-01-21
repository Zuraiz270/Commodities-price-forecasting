"""
Model Ensemble Utilities

Combine multiple models for improved prediction accuracy and robustness.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble."""
    
    combination_method: str = "mean"  # 'mean', 'median', 'weighted', 'stacking'
    weights: List[float] = None  # For weighted combination


class ModelEnsemble:
    """
    Ensemble of forecasting models.
    
    Supports multiple combination strategies:
    - Mean: Simple average of predictions
    - Median: Median of predictions
    - Weighted: Weighted average with learnable/fixed weights
    - Stacking: Meta-learner trained on model outputs
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: EnsembleConfig = None,
        model_names: List[str] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained model instances
            config: Ensemble configuration
            model_names: Optional names for each model
        """
        self.models = models
        self.config = config or EnsembleConfig()
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        
        # Set equal weights if not specified
        if self.config.weights is None:
            self.config.weights = [1.0 / len(models)] * len(models)
        
        # Validate
        if len(self.config.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate ensemble prediction.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with ensemble prediction and individual model predictions
        """
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model.forward(x)
                
                # Handle different output formats
                if isinstance(output, dict):
                    if "predictions" in output:
                        pred = output["predictions"]
                    elif "forecast" in output:
                        pred = output["forecast"]
                    else:
                        # TFT returns quantiles, take median
                        pred = output.get("predictions", output)
                        if pred.dim() == 3:  # [batch, horizon, quantiles]
                            pred = pred[..., 1]  # Take median quantile
                else:
                    pred = output
                
                all_predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(all_predictions, dim=0)  # [models, batch, horizon]
        
        # Combine based on method
        if self.config.combination_method == "mean":
            ensemble_pred = stacked.mean(dim=0)
        elif self.config.combination_method == "median":
            ensemble_pred = stacked.median(dim=0).values
        elif self.config.combination_method == "weighted":
            weights = torch.tensor(
                self.config.weights,
                device=stacked.device,
                dtype=stacked.dtype
            ).view(-1, 1, 1)
            ensemble_pred = (stacked * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown combination method: {self.config.combination_method}")
        
        return {
            "ensemble_prediction": ensemble_pred,
            "individual_predictions": {
                name: pred for name, pred in zip(self.model_names, all_predictions)
            }
        }
    
    def evaluate(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor
    ) -> pd.DataFrame:
        """
        Evaluate ensemble and individual models.
        
        Args:
            x: Input tensor
            y_true: Ground truth targets
            
        Returns:
            DataFrame with metrics for each model and ensemble
        """
        output = self.predict(x)
        
        metrics = []
        
        # Evaluate ensemble
        ensemble_pred = output["ensemble_prediction"]
        metrics.append({
            "model": "Ensemble",
            "MAE": self._mae(ensemble_pred, y_true),
            "RMSE": self._rmse(ensemble_pred, y_true),
            "MAPE": self._mape(ensemble_pred, y_true)
        })
        
        # Evaluate individual models
        for name, pred in output["individual_predictions"].items():
            metrics.append({
                "model": name,
                "MAE": self._mae(pred, y_true),
                "RMSE": self._rmse(pred, y_true),
                "MAPE": self._mape(pred, y_true)
            })
        
        return pd.DataFrame(metrics)
    
    def optimize_weights(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        metric: str = "mae"
    ) -> List[float]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            x_val: Validation input
            y_val: Validation targets
            metric: Metric to optimize ('mae', 'rmse', 'mape')
            
        Returns:
            Optimized weights
        """
        from scipy.optimize import minimize
        
        # Get individual predictions
        all_predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model.forward(x_val)
                if isinstance(output, dict):
                    pred = output.get("predictions", output.get("forecast"))
                    if pred.dim() == 3:
                        pred = pred[..., 1]
                else:
                    pred = output
                all_predictions.append(pred.numpy())
        
        stacked = np.stack(all_predictions, axis=0)
        y_np = y_val.numpy()
        
        metric_func = {
            "mae": lambda p: np.mean(np.abs(p - y_np)),
            "rmse": lambda p: np.sqrt(np.mean((p - y_np) ** 2)),
            "mape": lambda p: np.mean(np.abs((p - y_np) / (y_np + 1e-8))) * 100
        }[metric]
        
        def objective(weights):
            weights = np.array(weights) / np.sum(weights)  # Normalize
            combined = np.sum(stacked * weights.reshape(-1, 1, 1), axis=0)
            return metric_func(combined)
        
        # Optimize
        n_models = len(self.models)
        x0 = [1.0 / n_models] * n_models
        bounds = [(0, 1)] * n_models
        
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        
        optimized_weights = list(result.x / np.sum(result.x))
        self.config.weights = optimized_weights
        
        return optimized_weights
    
    @staticmethod
    def _mae(pred: torch.Tensor, true: torch.Tensor) -> float:
        return torch.mean(torch.abs(pred - true)).item()
    
    @staticmethod
    def _rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((pred - true) ** 2)).item()
    
    @staticmethod
    def _mape(pred: torch.Tensor, true: torch.Tensor) -> float:
        return (torch.mean(torch.abs((pred - true) / (true + 1e-8))) * 100).item()


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with a meta-learner.
    
    Trains a neural network to combine predictions from base models.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        output_size: int,
        hidden_size: int = 64
    ):
        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.n_models = len(base_models)
        
        # Freeze base models
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Meta-learner
        self.meta_learner = nn.Sequential(
            nn.Linear(self.n_models * output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through base models and meta-learner."""
        base_predictions = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                output = model.forward(x)
                if isinstance(output, dict):
                    pred = output.get("predictions", output.get("forecast"))
                    if pred.dim() == 3:
                        pred = pred[..., 1]
                else:
                    pred = output
                base_predictions.append(pred)
        
        # Concatenate base predictions
        stacked = torch.cat(base_predictions, dim=-1)
        
        # Meta-learner prediction
        meta_pred = self.meta_learner(stacked)
        
        return {
            "predictions": meta_pred,
            "base_predictions": base_predictions
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get stacking predictions."""
        with torch.no_grad():
            return self.forward(x)["predictions"]
