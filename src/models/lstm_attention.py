"""
LSTM with Attention Mechanism

Enhanced LSTM architecture with self-attention for improved
sequence modeling in time series forecasting.

Combines the sequential modeling strength of LSTMs with the
ability of attention mechanisms to capture long-range dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LSTMAttentionConfig:
    """Configuration for LSTM with Attention."""
    
    input_size: int = 1  # Number of input features
    hidden_size: int = 128  # LSTM hidden size
    num_layers: int = 2  # Number of LSTM layers
    output_size: int = 12  # Forecast horizon
    attention_heads: int = 4  # Number of attention heads
    dropout: float = 0.2
    bidirectional: bool = False
    use_layer_norm: bool = True


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_head)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class LSTMAttention(nn.Module):
    """
    LSTM with self-attention for time series forecasting.
    
    Architecture:
    1. Input projection
    2. Bidirectional/Unidirectional LSTM
    3. Layer normalization (optional)
    4. Multi-head self-attention
    5. Feed-forward network
    6. Output projection
    """
    
    def __init__(self, config: LSTMAttentionConfig):
        super().__init__()
        
        self.config = config
        
        # Account for bidirectional
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(lstm_output_size)
            self.layer_norm2 = nn.LayerNorm(lstm_output_size)
        else:
            self.layer_norm1 = nn.Identity()
            self.layer_norm2 = nn.Identity()
        
        # Self-attention
        self.attention = MultiHeadAttention(
            d_model=lstm_output_size,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size * 4, lstm_output_size)
        )
        
        # Output projection
        self.output_proj = nn.Linear(lstm_output_size, config.output_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
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
        # Input projection
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Layer norm
        lstm_out = self.layer_norm1(lstm_out)
        
        # Self-attention with residual
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        lstm_out = lstm_out + attn_out
        
        # Layer norm
        lstm_out = self.layer_norm2(lstm_out)
        
        # FFN with residual
        ffn_out = self.ffn(lstm_out)
        ffn_out = self.dropout(ffn_out)
        lstm_out = lstm_out + ffn_out
        
        # Use last timestep or pooled representation
        output = lstm_out[:, -1, :]  # Last timestep
        
        # Output projection
        predictions = self.output_proj(output)
        
        result = {"predictions": predictions}
        
        if return_attention:
            result["attention_weights"] = attention_weights
        
        return result
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get point predictions."""
        with torch.no_grad():
            output = self.forward(x)
            return output["predictions"]
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        with torch.no_grad():
            output = self.forward(x, return_attention=True)
            return output["attention_weights"]


class TemporalConvLSTM(nn.Module):
    """
    Hybrid Temporal Convolutional + LSTM architecture.
    
    Uses 1D convolutions to extract local patterns before
    feeding to LSTM for sequential modeling.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        output_size: int = 12,
        num_conv_layers: int = 3,
        kernel_size: int = 3,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Temporal convolution blocks
        conv_layers = []
        in_channels = input_size
        
        for i in range(num_conv_layers):
            out_channels = hidden_size // (2 ** (num_conv_layers - 1 - i))
            out_channels = max(out_channels, 16)
            
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_out_size = in_channels
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.conv_out_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            
        Returns:
            Dictionary with predictions
        """
        # Conv expects [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        conv_out = self.conv_layers(x)
        
        # Back to [batch, seq_len, channels]
        conv_out = conv_out.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # Use last timestep
        output = lstm_out[:, -1, :]
        
        # Project to output
        predictions = self.output_proj(output)
        
        return {"predictions": predictions}
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get point predictions."""
        with torch.no_grad():
            return self.forward(x)["predictions"]
