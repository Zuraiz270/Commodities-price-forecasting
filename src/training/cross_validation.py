"""
Time Series Cross-Validation

Proper train/validation/test splitting for time series data.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CVConfig:
    """Cross-validation configuration."""
    
    n_splits: int = 5
    test_size: int = 12  # Size of test set (months for commodity data)
    gap: int = 0  # Gap between train and test to avoid leakage


class TimeSeriesCV:
    """
    Time Series Cross-Validation with walk-forward validation.
    
    Methods:
    - Walk-forward: Train on past, test on future
    - Expanding window: Growing training set
    - Sliding window: Fixed-size training window
    """
    
    def __init__(self, config: CVConfig = None):
        self.config = config or CVConfig()
    
    def walk_forward_split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        n_splits: int = None,
        test_size: int = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Walk-forward cross-validation.
        
        Each split: train on all past data, test on future window.
        
        Args:
            X: Feature array
            y: Target array (optional, same splits applied)
            n_splits: Number of splits
            test_size: Size of each test set
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_splits = n_splits or self.config.n_splits
        test_size = test_size or self.config.test_size
        
        n_samples = len(X)
        min_train_size = n_samples - (n_splits * test_size) - (n_splits - 1) * self.config.gap
        
        if min_train_size < test_size:
            raise ValueError(f"Not enough samples for {n_splits} splits with test_size={test_size}")
        
        for i in range(n_splits):
            # Calculate split points
            test_end = n_samples - i * (test_size + self.config.gap)
            test_start = test_end - test_size
            train_end = test_start - self.config.gap
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def expanding_window_split(
        self,
        X: np.ndarray,
        initial_train_size: int = None,
        step_size: int = None,
        test_size: int = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Expanding window cross-validation.
        
        Training set grows with each split.
        
        Args:
            X: Feature array
            initial_train_size: Initial training set size
            step_size: How much to expand training set each split
            test_size: Size of test set
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size = test_size or self.config.test_size
        step_size = step_size or test_size
        initial_train_size = initial_train_size or max(n_samples // 3, test_size * 2)
        
        train_end = initial_train_size
        
        while train_end + test_size + self.config.gap <= n_samples:
            test_start = train_end + self.config.gap
            test_end = min(test_start + test_size, n_samples)
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            train_end += step_size
    
    def sliding_window_split(
        self,
        X: np.ndarray,
        train_size: int,
        test_size: int = None,
        step_size: int = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Sliding window cross-validation.
        
        Fixed-size training window slides through data.
        
        Args:
            X: Feature array
            train_size: Fixed training set size
            test_size: Size of test set
            step_size: How much to slide window each split
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size = test_size or self.config.test_size
        step_size = step_size or test_size
        
        train_start = 0
        
        while train_start + train_size + test_size + self.config.gap <= n_samples:
            train_end = train_start + train_size
            test_start = train_end + self.config.gap
            test_end = test_start + test_size
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            train_start += step_size
    
    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[np.ndarray, ...]:
        """
        Simple chronological train/val/test split.
        
        Args:
            X: Feature array
            y: Target array (optional)
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (X_train, X_val, X_test, [y_train, y_val, y_test])
        """
        n_samples = len(X)
        
        test_size = int(n_samples * test_ratio)
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size - test_size
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        if y is not None:
            y_train = y[:train_size]
            y_val = y[train_size:train_size + val_size]
            y_test = y[train_size + val_size:]
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        return X_train, X_val, X_test
    
    def get_cv_scores(
        self,
        model_class,
        model_params: dict,
        X: np.ndarray,
        y: np.ndarray,
        cv_method: str = "walk_forward",
        fit_params: dict = None
    ) -> pd.DataFrame:
        """
        Compute cross-validation scores.
        
        Args:
            model_class: Model class to instantiate
            model_params: Parameters for model
            X: Feature array
            y: Target array
            cv_method: 'walk_forward', 'expanding', or 'sliding'
            fit_params: Additional parameters for fit
            
        Returns:
            DataFrame with CV scores per fold
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        fit_params = fit_params or {}
        
        # Get splits
        if cv_method == "walk_forward":
            splits = list(self.walk_forward_split(X, y))
        elif cv_method == "expanding":
            splits = list(self.expanding_window_split(X))
        elif cv_method == "sliding":
            splits = list(self.sliding_window_split(X, train_size=len(X) // 2))
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"Fold {fold_idx + 1}/{len(splits)}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create fresh model
            model = model_class(**model_params)
            
            # Train
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train, **fit_params)
            else:
                # PyTorch model - simple training
                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                y_train_t = torch.tensor(y_train, dtype=torch.float32)
                
                dataset = TensorDataset(X_train_t, y_train_t)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.MSELoss()
                
                model.train()
                for epoch in range(50):
                    for batch_x, batch_y in loader:
                        optimizer.zero_grad()
                        output = model(batch_x)
                        if isinstance(output, dict):
                            pred = output.get("predictions", output.get("forecast"))
                            if pred.dim() == 3:
                                pred = pred[..., 1]
                        else:
                            pred = output
                        loss = criterion(pred, batch_y)
                        loss.backward()
                        optimizer.step()
            
            # Predict
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
            else:
                model.eval()
                with torch.no_grad():
                    X_test_t = torch.tensor(X_test, dtype=torch.float32)
                    output = model(X_test_t)
                    if isinstance(output, dict):
                        predictions = output.get("predictions", output.get("forecast"))
                        if predictions.dim() == 3:
                            predictions = predictions[..., 1]
                    else:
                        predictions = output
                    predictions = predictions.numpy()
            
            # Metrics
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            mape = np.mean(np.abs((predictions - y_test) / (y_test + 1e-8))) * 100
            
            results.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape
            })
        
        return pd.DataFrame(results)
