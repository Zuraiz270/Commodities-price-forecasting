"""
N-HiTS Model (Neural Hierarchical Interpolation for Time Series)

Enhanced version of N-BEATS with multi-rate data sampling and
hierarchical interpolation for improved long-horizon forecasting.

Based on: "N-HiTS: Neural Hierarchical Interpolation for Time
Series Forecasting" (Challu et al., 2022)

Key improvements over N-BEATS:
- Multi-rate data sampling captures short and long patterns
- Hierarchical interpolation reduces computational cost
- Better performance on long forecast horizons
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class NHITSConfig:
    """Configuration for N-HiTS model."""
    
    input_size: int = 60  # Lookback window
    output_size: int = 12  # Forecast horizon
    hidden_size: int = 256  # Hidden layer size
    num_stacks: int = 3  # Number of stacks
    num_blocks_per_stack: int = 1  # Blocks per stack
    pooling_sizes: List[int] = None  # Multi-rate sampling
    interpolation_mode: str = "linear"  # 'linear' or 'nearest'
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.pooling_sizes is None:
            # Default: progressively larger pooling for each stack
            self.pooling_sizes = [1, 2, 4][:self.num_stacks]


class NHITSBlock(nn.Module):
    """
    N-HiTS block with multi-rate sampling and interpolation.
    
    Each block operates on a downsampled version of the input
    and interpolates back to the original resolution.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        pooling_size: int = 1,
        n_theta: int = None,
        interpolation_mode: str = "linear",
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.pooling_size = pooling_size
        self.interpolation_mode = interpolation_mode
        
        # Downsampled sizes
        self.pooled_input_size = max(1, input_size // pooling_size)
        self.pooled_output_size = max(1, output_size // pooling_size)
        
        # Theta sizes (basis expansion coefficients)
        self.n_theta_backcast = n_theta or self.pooled_input_size
        self.n_theta_forecast = n_theta or self.pooled_output_size
        
        # MaxPool for multi-rate sampling
        if pooling_size > 1:
            self.pooling = nn.MaxPool1d(
                kernel_size=pooling_size,
                stride=pooling_size
            )
        else:
            self.pooling = None
        
        # FC stack
        layers = []
        for i in range(num_layers):
            in_features = self.pooled_input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta projections
        self.theta_backcast = nn.Linear(hidden_size, self.n_theta_backcast)
        self.theta_forecast = nn.Linear(hidden_size, self.n_theta_forecast)
    
    def interpolate(
        self,
        x: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        """Interpolate tensor to target size."""
        if x.size(-1) == target_size:
            return x
        
        # Add dimension for interpolation
        x_3d = x.unsqueeze(1)
        
        interpolated = nn.functional.interpolate(
            x_3d,
            size=target_size,
            mode=self.interpolation_mode
        )
        
        return interpolated.squeeze(1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with multi-rate sampling.
        
        Returns:
            Tuple of (backcast, forecast) at original resolution
        """
        # Multi-rate sampling (downsample)
        if self.pooling is not None:
            # Add channel dimension for pooling
            pooled = self.pooling(x.unsqueeze(1)).squeeze(1)
        else:
            pooled = x
        
        # FC processing
        hidden = self.fc_stack(pooled)
        
        # Generate theta coefficients
        theta_back = self.theta_backcast(hidden)
        theta_fore = self.theta_forecast(hidden)
        
        # Hierarchical interpolation (upsample)
        backcast = self.interpolate(theta_back, self.input_size)
        forecast = self.interpolate(theta_fore, self.output_size)
        
        return backcast, forecast


class NHITSModel(nn.Module):
    """
    N-HiTS model for commodity price forecasting.
    
    Combines multi-rate data sampling with hierarchical interpolation
    for superior long-horizon forecasting performance.
    """
    
    def __init__(self, config: NHITSConfig):
        super().__init__()
        
        self.config = config
        
        # Validate pooling sizes
        if len(config.pooling_sizes) != config.num_stacks:
            raise ValueError(
                f"pooling_sizes length ({len(config.pooling_sizes)}) "
                f"must match num_stacks ({config.num_stacks})"
            )
        
        # Build stacks with different pooling rates
        self.stacks = nn.ModuleList()
        
        for stack_idx in range(config.num_stacks):
            blocks = nn.ModuleList([
                NHITSBlock(
                    input_size=config.input_size,
                    output_size=config.output_size,
                    hidden_size=config.hidden_size,
                    pooling_size=config.pooling_sizes[stack_idx],
                    interpolation_mode=config.interpolation_mode,
                    dropout=config.dropout
                )
                for _ in range(config.num_blocks_per_stack)
            ])
            self.stacks.append(blocks)
    
    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, input_size] or [batch, 1, input_size]
            return_components: Whether to return per-stack forecasts
            
        Returns:
            Dictionary with forecast and optionally stack components
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
        
        if return_components:
            result["stack_forecasts"] = stack_forecasts
            result["residuals"] = residuals
        
        return result
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get point predictions."""
        with torch.no_grad():
            output = self.forward(x)
            return output["forecast"]
    
    def get_multi_scale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get multi-scale representations from each stack.
        
        Useful for understanding what patterns each scale captures.
        """
        with torch.no_grad():
            output = self.forward(x, return_components=True)
            return output["stack_forecasts"]


def create_nhits_from_neuralforecast(
    horizon: int,
    input_size: int = 60,
    n_pool_kernel_size: List[int] = None
) -> Any:
    """
    Create N-HiTS model using neuralforecast library.
    
    Args:
        horizon: Forecast horizon
        input_size: Lookback window
        n_pool_kernel_size: Pooling kernel sizes per stack
        
    Returns:
        neuralforecast NHITS model instance
    """
    try:
        from neuralforecast.models import NHITS
        
        n_pool_kernel_size = n_pool_kernel_size or [2, 2, 2]
        
        model = NHITS(
            h=horizon,
            input_size=input_size,
            n_pool_kernel_size=n_pool_kernel_size,
            n_freq_downsample=[4, 2, 1],
            mlp_units=[[256, 256], [256, 256], [256, 256]],
            max_steps=1000,
            early_stop_patience_steps=50,
            val_check_steps=50
        )
        
        return model
        
    except ImportError:
        raise ImportError(
            "neuralforecast not installed. Install with: pip install neuralforecast"
        )
