"""
Data Loading Utilities

Dataset classes and data loading utilities for commodity price forecasting.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Union, Dict
from pathlib import Path


class CommodityDataset(Dataset):
    """
    PyTorch Dataset for commodity price time series.
    
    Handles windowed sequence creation for model training.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray = None,
        context_length: int = 60,
        prediction_horizon: int = 12,
        stride: int = 1
    ):
        """
        Initialize dataset.
        
        Args:
            data: Feature array [n_samples, n_features]
            targets: Target array (if None, uses last column of data)
            context_length: Input sequence length
            prediction_horizon: Output sequence length
            stride: Step size between windows
        """
        self.data = data
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        if targets is None:
            self.targets = data[:, -1] if data.ndim > 1 else data
        else:
            self.targets = targets
        
        # Calculate valid indices
        self.n_samples = len(data)
        self.valid_indices = list(range(
            context_length,
            self.n_samples - prediction_horizon + 1,
            stride
        ))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.context_length
        target_end = end_idx + self.prediction_horizon
        
        # Input sequence
        x = self.data[start_idx:end_idx]
        
        # Target sequence
        y = self.targets[end_idx:target_end]
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


class DataLoader:
    """
    Data loading and preprocessing utilities.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path] = None,
        commodity: str = "aluminum"
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
            commodity: Commodity type ('aluminum', 'copper', 'zinc')
        """
        self.data_path = Path(data_path) if data_path else None
        self.commodity = commodity
        self.raw_data = None
        self.processed_data = None
    
    def load_csv(
        self,
        filepath: Union[str, Path],
        date_col: str = "Date",
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            date_col: Name of date column
            parse_dates: Whether to parse dates
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(
            filepath,
            parse_dates=[date_col] if parse_dates and date_col else False
        )
        
        if date_col and date_col in df.columns:
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
        
        self.raw_data = df
        return df
    
    def load_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        date_col: str = "Date"
    ) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            date_col: Name of date column
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
        
        self.raw_data = df
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame = None,
        target_col: str = "Price",
        fill_method: str = "ffill",
        normalize: bool = True,
        normalization_method: str = "minmax"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess data for modeling.
        
        Args:
            df: DataFrame to preprocess (uses self.raw_data if None)
            target_col: Name of target column
            fill_method: Method for filling missing values
            normalize: Whether to normalize data
            normalization_method: 'minmax' or 'zscore'
            
        Returns:
            Tuple of (processed_array, scaling_params)
        """
        df = df if df is not None else self.raw_data
        
        if df is None:
            raise ValueError("No data loaded. Call load_csv() or load_excel() first.")
        
        df = df.copy()
        
        # Handle missing values
        if fill_method == "ffill":
            df = df.fillna(method="ffill")
        elif fill_method == "bfill":
            df = df.fillna(method="bfill")
        elif fill_method == "interpolate":
            df = df.interpolate(method="linear")
        
        # Drop any remaining NaN
        df = df.dropna()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values
        
        # Normalize
        scaling_params = {}
        
        if normalize:
            if normalization_method == "minmax":
                data_min = data.min(axis=0)
                data_max = data.max(axis=0)
                data = (data - data_min) / (data_max - data_min + 1e-8)
                scaling_params = {
                    "method": "minmax",
                    "min": data_min,
                    "max": data_max
                }
            elif normalization_method == "zscore":
                data_mean = data.mean(axis=0)
                data_std = data.std(axis=0)
                data = (data - data_mean) / (data_std + 1e-8)
                scaling_params = {
                    "method": "zscore",
                    "mean": data_mean,
                    "std": data_std
                }
        
        self.processed_data = data
        self.column_names = list(numeric_cols)
        self.scaling_params = scaling_params
        
        return data, scaling_params
    
    def inverse_transform(
        self,
        data: np.ndarray,
        scaling_params: Dict = None,
        column_idx: int = None
    ) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data
            scaling_params: Scaling parameters (uses stored if None)
            column_idx: Index of column to inverse transform (all if None)
            
        Returns:
            Original scale data
        """
        params = scaling_params or self.scaling_params
        
        if not params:
            return data
        
        if params["method"] == "minmax":
            if column_idx is not None:
                data_min = params["min"][column_idx]
                data_max = params["max"][column_idx]
            else:
                data_min = params["min"]
                data_max = params["max"]
            return data * (data_max - data_min) + data_min
        
        elif params["method"] == "zscore":
            if column_idx is not None:
                data_mean = params["mean"][column_idx]
                data_std = params["std"][column_idx]
            else:
                data_mean = params["mean"]
                data_std = params["std"]
            return data * data_std + data_mean
        
        return data
    
    def create_dataloaders(
        self,
        data: np.ndarray = None,
        targets: np.ndarray = None,
        context_length: int = 60,
        prediction_horizon: int = 12,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        shuffle_train: bool = True
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        Create train/val/test DataLoaders.
        
        Args:
            data: Feature array (uses processed_data if None)
            targets: Target array
            context_length: Input sequence length
            prediction_horizon: Output sequence length
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            batch_size: Batch size
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data = data if data is not None else self.processed_data
        
        if data is None:
            raise ValueError("No data available. Call preprocess() first.")
        
        n_samples = len(data)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Split data chronologically
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        if targets is not None:
            train_targets = targets[:train_size]
            val_targets = targets[train_size:train_size + val_size]
            test_targets = targets[train_size + val_size:]
        else:
            train_targets = val_targets = test_targets = None
        
        # Create datasets
        train_dataset = CommodityDataset(
            train_data, train_targets,
            context_length, prediction_horizon
        )
        val_dataset = CommodityDataset(
            val_data, val_targets,
            context_length, prediction_horizon
        )
        test_dataset = CommodityDataset(
            test_data, test_targets,
            context_length, prediction_horizon
        )
        
        # Create loaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
