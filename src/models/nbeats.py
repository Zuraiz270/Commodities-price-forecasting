"""
N-BEATS Model (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)

Pure deep learning architecture that achieves state-of-the-art results
without time series specific preprocessing.

Based on: "N-BEATS: Neural basis expansion analysis for interpretable
time series forecasting" (Oreshkin et al., 2019)

This module provides a wrapper around the neuralforecast library implementation
for easy integration with the commodity forecasting pipeline.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class NBEATSConfig:
    """Configuration for N-BEATS model."""
    
    input_size: int = 60  # Lookback window
    output_size: int = 12  # Forecast horizon
    hidden_size: int = 256  # Hidden layer size
    num_blocks: int = 3  # Number of blocks per stack
    num_stacks: int = 2  # Number of stacks
    block_type: str = "generic"  # 'generic', 'trend', or 'seasonality'
    dropout: float = 0.1
    learning_rate: float = 1e-3
    
    # For interpretable variant
    trend_blocks: int = 3
    seasonality_blocks: int = 3
    polynomial_degree: int = 2  # For trend basis
    harmonics: int = 1  # For seasonality basis


class NBEATSBlock(nn.Module):
    """
    Basic N-BEATS block.
    
    Each block has a multi-layer FC stack that produces both
    backcast (reconstruction of input) and forecast outputs.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # FC stack
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta projections for backcast and forecast
        self.theta_backcast = nn.Linear(hidden_size, input_size)
        self.theta_forecast = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            Tuple of (backcast, forecast)
        """
        hidden = self.fc_stack(x)
        backcast = self.theta_backcast(hidden)
        forecast = self.theta_forecast(hidden)
        return backcast, forecast


class NBEATSTrendBlock(nn.Module):
    """
    Interpretable N-BEATS block for trend modeling.
    
    Uses polynomial basis functions for interpretable trend extraction.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        polynomial_degree: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.polynomial_degree = polynomial_degree
        
        # FC stack
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta for polynomial coefficients
        self.theta = nn.Linear(hidden_size, polynomial_degree + 1)
        
        # Create basis matrices (polynomial terms)
        self._create_basis(input_size, output_size)
    
    def _create_basis(self, input_size: int, output_size: int):
        """Create polynomial basis matrices."""
        # Backcast basis
        t_back = torch.linspace(0, 1, input_size).unsqueeze(0)
        back_basis = torch.stack([
            t_back ** i for i in range(self.polynomial_degree + 1)
        ], dim=1).squeeze(-1)
        self.register_buffer('back_basis', back_basis)
        
        # Forecast basis
        t_fore = torch.linspace(0, 1, output_size).unsqueeze(0)
        fore_basis = torch.stack([
            t_fore ** i for i in range(self.polynomial_degree + 1)
        ], dim=1).squeeze(-1)
        self.register_buffer('fore_basis', fore_basis)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass with trend basis expansion."""
        hidden = self.fc_stack(x)
        theta = self.theta(hidden)  # [batch, degree+1]
        
        # Expand with polynomial basis
        backcast = torch.einsum('bp,pt->bt', theta, self.back_basis)
        forecast = torch.einsum('bp,pt->bt', theta, self.fore_basis)
        
        return backcast, forecast


class NBEATSSeasonalityBlock(nn.Module):
    """
    Interpretable N-BEATS block for seasonality modeling.
    
    Uses Fourier basis functions for interpretable seasonality extraction.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        harmonics: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.harmonics = harmonics
        
        # FC stack
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta for Fourier coefficients (sin and cos for each harmonic)
        num_coeffs = 2 * harmonics
        self.theta = nn.Linear(hidden_size, num_coeffs)
        
        # Create Fourier basis matrices
        self._create_basis(input_size, output_size)
    
    def _create_basis(self, input_size: int, output_size: int):
        """Create Fourier basis matrices."""
        def fourier_basis(length: int) -> torch.Tensor:
            t = torch.linspace(0, 2 * np.pi, length).unsqueeze(0)
            basis = []
            for h in range(1, self.harmonics + 1):
                basis.append(torch.sin(h * t))
                basis.append(torch.cos(h * t))
            return torch.stack(basis, dim=1).squeeze(-1)
        
        self.register_buffer('back_basis', fourier_basis(input_size))
        self.register_buffer('fore_basis', fourier_basis(output_size))
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass with Fourier basis expansion."""
        hidden = self.fc_stack(x)
        theta = self.theta(hidden)  # [batch, 2*harmonics]
        
        # Expand with Fourier basis
        backcast = torch.einsum('bp,pt->bt', theta, self.back_basis)
        forecast = torch.einsum('bp,pt->bt', theta, self.fore_basis)
        
        return backcast, forecast


class NBEATSModel(nn.Module):
    """
    N-BEATS model for commodity price forecasting.
    
    Supports both generic (black-box) and interpretable configurations.
    
    Generic: Pure deep learning with no assumptions about time series structure
    Interpretable: Separate trend and seasonality stacks with basis functions
    """
    
    def __init__(self, config: NBEATSConfig):
        super().__init__()
        
        self.config = config
        
        if config.block_type == "generic":
            self._build_generic_stacks()
        else:
            self._build_interpretable_stacks()
    
    def _build_generic_stacks(self):
        """Build generic N-BEATS stacks."""
        self.stacks = nn.ModuleList()
        
        for _ in range(self.config.num_stacks):
            blocks = nn.ModuleList([
                NBEATSBlock(
                    input_size=self.config.input_size,
                    output_size=self.config.output_size,
                    hidden_size=self.config.hidden_size,
                    dropout=self.config.dropout
                )
                for _ in range(self.config.num_blocks)
            ])
            self.stacks.append(blocks)
    
    def _build_interpretable_stacks(self):
        """Build interpretable N-BEATS stacks (trend + seasonality)."""
        self.stacks = nn.ModuleList()
        
        # Trend stack
        trend_blocks = nn.ModuleList([
            NBEATSTrendBlock(
                input_size=self.config.input_size,
                output_size=self.config.output_size,
                hidden_size=self.config.hidden_size,
                polynomial_degree=self.config.polynomial_degree,
                dropout=self.config.dropout
            )
            for _ in range(self.config.trend_blocks)
        ])
        self.stacks.append(trend_blocks)
        
        # Seasonality stack
        seasonality_blocks = nn.ModuleList([
            NBEATSSeasonalityBlock(
                input_size=self.config.input_size,
                output_size=self.config.output_size,
                hidden_size=self.config.hidden_size,
                harmonics=self.config.harmonics,
                dropout=self.config.dropout
            )
            for _ in range(self.config.seasonality_blocks)
        ])
        self.stacks.append(seasonality_blocks)
    
    def forward(
        self,
        x: torch.Tensor,
        return_decomposition: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, input_size] or [batch, 1, input_size]
            return_decomposition: Whether to return per-stack forecasts
            
        Returns:
            Dictionary with forecast and optionally stack decompositions
        """
        # Handle 3D input
        if x.dim() == 3:
            x = x.squeeze(1)
        
        residuals = x
        forecast = torch.zeros(x.size(0), self.config.output_size, device=x.device)
        
        stack_forecasts = []
        
        for stack in self.stacks:
            stack_forecast = torch.zeros_like(forecast)
            
            for block in stack:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                stack_forecast = stack_forecast + block_forecast
            
            forecast = forecast + stack_forecast
            stack_forecasts.append(stack_forecast)
        
        result = {"forecast": forecast}
        
        if return_decomposition:
            result["stack_forecasts"] = stack_forecasts
            if self.config.block_type != "generic":
                result["trend"] = stack_forecasts[0]
                result["seasonality"] = stack_forecasts[1]
        
        return result
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get point predictions."""
        with torch.no_grad():
            output = self.forward(x)
            return output["forecast"]
    
    def decompose(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose time series into trend and seasonality.
        
        Only available for interpretable configuration.
        """
        if self.config.block_type == "generic":
            raise ValueError("Decomposition only available for interpretable N-BEATS")
        
        with torch.no_grad():
            output = self.forward(x, return_decomposition=True)
            return {
                "trend": output["trend"],
                "seasonality": output["seasonality"],
                "forecast": output["forecast"]
            }


def create_nbeats_from_neuralforecast(
    horizon: int,
    input_size: int = 60,
    stack_types: List[str] = None
) -> Any:
    """
    Create N-BEATS model using neuralforecast library.
    
    This provides a production-ready implementation with
    built-in training utilities.
    
    Args:
        horizon: Forecast horizon
        input_size: Lookback window
        stack_types: List of stack types ('generic', 'trend', 'seasonality')
        
    Returns:
        neuralforecast NBEATS model instance
    """
    try:
        from neuralforecast.models import NBEATS
        
        stack_types = stack_types or ["trend", "seasonality"]
        
        model = NBEATS(
            h=horizon,
            input_size=input_size,
            stack_types=stack_types,
            n_blocks=[3, 3],
            mlp_units=[[256, 256], [256, 256]],
            max_steps=1000,
            early_stop_patience_steps=50,
            val_check_steps=50
        )
        
        return model
        
    except ImportError:
        raise ImportError(
            "neuralforecast not installed. Install with: pip install neuralforecast"
        )
