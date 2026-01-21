"""
Temporal Fusion Transformer (TFT) Model

State-of-the-art architecture for multi-horizon time series forecasting.
Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon 
Time Series Forecasting" (Lim et al., 2021)

Key features:
- Variable selection networks for automatic feature importance
- Interpretable multi-head attention
- Gating mechanisms for skipping unused components
- Static covariate encoders
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    
    input_size: int = 1  # Number of input features
    hidden_size: int = 64  # Hidden layer size
    attention_heads: int = 4  # Number of attention heads
    dropout: float = 0.1  # Dropout rate
    num_encoder_layers: int = 2  # Number of encoder layers
    context_length: int = 60  # Input sequence length
    prediction_horizon: int = 12  # Output sequence length
    num_static_features: int = 0  # Static covariates
    num_time_varying_known: int = 0  # Known future inputs
    num_time_varying_unknown: int = 1  # Unknown future inputs (targets)
    quantiles: List[float] = None  # Quantiles for probabilistic forecasting
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - core building block of TFT.
    
    Provides non-linear processing with gating mechanism for
    flexible information flow.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()
        
        output_size = output_size or input_size
        self.context_size = context_size
        
        # Main layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        
        # Context projection if available
        if context_size:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)
        
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Gating
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection projection if sizes differ
        self.skip_proj = None
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass with optional context."""
        # Store for skip connection
        residual = x
        if self.skip_proj:
            residual = self.skip_proj(residual)
        
        # Main processing
        hidden = self.fc1(x)
        
        # Add context if available
        if context is not None and self.context_size:
            hidden = hidden + self.context_proj(context)
        
        hidden = self.elu(hidden)
        
        # Generate gate from the hidden state before fc2
        gate = self.sigmoid(self.gate(hidden))
        
        # Project to output size
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating
        hidden = gate * hidden
        
        # Residual connection and normalization
        output = self.layer_norm(hidden + residual)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - automatically selects relevant features.
    
    Provides interpretable feature importance through learned weights.
    """
    
    def __init__(
        self,
        input_size: int,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()
        
        self.num_features = num_features
        
        # GRN for each input feature
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
            for _ in range(num_features)
        ])
        
        # Softmax weights for variable selection
        self.weight_grn = GatedResidualNetwork(
            input_size=num_features * hidden_size,
            hidden_size=hidden_size,
            output_size=num_features,
            dropout=dropout,
            context_size=context_size
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning selected features and weights.
        
        Returns:
            Tuple of (selected_features, variable_weights)
        """
        # Process each feature
        processed_features = []
        for i, grn in enumerate(self.feature_grns):
            feat = grn(x[..., i:i+1] if x.dim() == 3 else x[:, i:i+1])
            processed_features.append(feat)
        
        # Stack processed features
        stacked = torch.stack(processed_features, dim=-1)  # [B, T, H, F]
        
        # Compute selection weights
        flattened = stacked.view(*stacked.shape[:-2], -1)  # [B, T, H*F]
        weights = self.weight_grn(flattened, context)
        weights = self.softmax(weights)  # [B, T, F]
        
        # Apply weights
        selected = (stacked * weights.unsqueeze(-2)).sum(dim=-1)
        
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for TFT.
    
    Modified attention that shares values across heads for interpretability.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Query, Key projections per head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        
        # Shared value projection
        self.W_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning output and attention weights.
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Project Q, K, V
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        # Average attention across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer for commodity price forecasting.
    
    State-of-the-art architecture providing:
    - Interpretable variable selection
    - Temporal attention visualization
    - Quantile predictions for uncertainty
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            config.context_length + config.prediction_horizon,
            config.hidden_size
        )
        
        # Variable selection (simplified for commodity forecasting)
        self.variable_selection = VariableSelectionNetwork(
            input_size=1,
            num_features=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_encoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0
        )
        
        # Gated skip connection
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        
        # Self-attention
        self.attention = InterpretableMultiHeadAttention(
            d_model=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )
        
        # Post-attention processing
        self.post_attention = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Output projection (quantile outputs)
        self.output_proj = nn.Linear(
            config.hidden_size,
            config.prediction_horizon * len(config.quantiles)
        )
        
        self._init_weights()
    
    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        selected, var_weights = self.variable_selection(x)
        
        # Add positional encoding
        selected = selected + self.pos_encoding[:, :seq_len, :]
        
        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected)
        
        # Gated skip connection
        gate = self.gate(lstm_out)
        enriched = gate * lstm_out + (1 - gate) * selected
        
        # Self-attention
        attn_out, attention_weights = self.attention(
            enriched, enriched, enriched
        )
        
        # Post-attention processing
        output = self.post_attention(attn_out + enriched)
        
        # Take last timestep for prediction
        output = output[:, -1, :]
        
        # Project to quantile predictions
        predictions = self.output_proj(output)
        predictions = predictions.view(
            batch_size,
            self.config.prediction_horizon,
            len(self.config.quantiles)
        )
        
        result = {
            "predictions": predictions,
            "variable_weights": var_weights
        }
        
        if return_attention:
            result["attention_weights"] = attention_weights
        
        return result
    
    def predict(
        self,
        x: torch.Tensor,
        quantile: float = 0.5
    ) -> torch.Tensor:
        """
        Get point predictions for a specific quantile.
        
        Args:
            x: Input tensor
            quantile: Quantile to return (default 0.5 for median)
            
        Returns:
            Point predictions tensor
        """
        with torch.no_grad():
            output = self.forward(x)
            predictions = output["predictions"]
            
            # Find index of requested quantile
            quantile_idx = self.config.quantiles.index(quantile)
            return predictions[..., quantile_idx]
    
    def get_variable_importance(
        self,
        x: torch.Tensor
    ) -> pd.DataFrame:
        """
        Get variable importance scores.
        
        Args:
            x: Input tensor
            
        Returns:
            DataFrame with variable importance scores
        """
        with torch.no_grad():
            output = self.forward(x, return_attention=True)
            var_weights = output["variable_weights"]
            
            # Average across batch and time
            importance = var_weights.mean(dim=(0, 1)).cpu().numpy()
            
            return pd.DataFrame({
                "feature_idx": range(len(importance)),
                "importance": importance
            })
