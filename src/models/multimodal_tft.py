"""
Multimodal Temporal Fusion Transformer

TFT extended with text encoder for multimodal commodity price forecasting.
Combines price history with news sentiment/embeddings using late fusion.

Based on research: "Event-Based forecasting where LLMs extract structured 
events from news to aid numerical models."
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .tft import TFTModel, TFTConfig, GatedResidualNetwork


@dataclass
class MultimodalTFTConfig:
    """Configuration for Multimodal TFT."""
    
    # Price model config
    price_input_size: int = 1
    hidden_size: int = 64
    attention_heads: int = 4
    dropout: float = 0.1
    num_encoder_layers: int = 2
    context_length: int = 60
    prediction_horizon: int = 12
    quantiles: List[float] = None
    
    # Text config
    text_embedding_dim: int = 384  # Sentence transformer output
    text_hidden_size: int = 64
    use_text_attention: bool = True
    
    # Fusion config
    fusion_method: str = "late"  # 'late', 'early', 'attention'
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class TextEncoder(nn.Module):
    """
    Encode daily text embeddings to match price feature dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode text embeddings.
        
        Args:
            x: Text embeddings [batch, seq_len, text_dim]
            
        Returns:
            Encoded [batch, seq_len, output_dim]
        """
        return self.encoder(x)


class TextPriceAttention(nn.Module):
    """
    Attention mechanism for combining text and price features.
    
    Allows price encoder to selectively attend to relevant news.
    """
    
    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project to common dimension
        self.price_proj = nn.Linear(price_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output gate (decide how much text info to use)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        price_features: torch.Tensor,
        text_features: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-modal attention.
        
        Args:
            price_features: [batch, price_seq, price_dim]
            text_features: [batch, text_seq, text_dim]
            return_weights: Return attention weights
            
        Returns:
            Fused features and optionally attention weights
        """
        # Project to common space
        price_proj = self.price_proj(price_features)
        text_proj = self.text_proj(text_features)
        
        # Price attends to text
        attn_out, attn_weights = self.attention(
            query=price_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Gated residual
        gate_input = torch.cat([price_proj, attn_out], dim=-1)
        gate = self.gate(gate_input)
        
        output = gate * attn_out + (1 - gate) * price_proj
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        if return_weights:
            return output, attn_weights
        return output, None


class MultimodalTFT(nn.Module):
    """
    Multimodal Temporal Fusion Transformer.
    
    Combines:
    1. Price/numeric time series (via TFT backbone)
    2. Text embeddings from news (via TextEncoder)
    3. Cross-modal attention for fusion
    
    Architecture:
    - Price branch: Standard TFT layers
    - Text branch: Linear encoder for embeddings
    - Fusion: Late fusion with cross-modal attention
    """
    
    def __init__(self, config: MultimodalTFTConfig):
        super().__init__()
        
        self.config = config
        
        # Price input projection
        self.price_input_proj = nn.Linear(
            config.price_input_size,
            config.hidden_size
        )
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            config.context_length + config.prediction_horizon,
            config.hidden_size
        )
        
        # LSTM encoder for price
        self.price_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_encoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            input_dim=config.text_embedding_dim,
            hidden_dim=config.text_hidden_size,
            output_dim=config.hidden_size,
            dropout=config.dropout
        )
        
        # Cross-modal attention
        if config.use_text_attention:
            self.text_price_attention = TextPriceAttention(
                price_dim=config.hidden_size,
                text_dim=config.hidden_size,
                hidden_dim=config.hidden_size,
                num_heads=config.attention_heads,
                dropout=config.dropout
            )
        else:
            self.text_price_attention = None
        
        # Self-attention for temporal modeling
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Post-attention processing
        self.post_attention = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Fusion layer
        if config.fusion_method == "late":
            # Concatenate and project
            self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.fusion = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Output projection
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
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        price_input: torch.Tensor,
        text_input: torch.Tensor = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            price_input: Price features [batch, seq_len, price_features]
            text_input: Text embeddings [batch, seq_len, text_dim]
            return_attention: Return attention weights
            
        Returns:
            Dictionary with predictions and optionally attention
        """
        batch_size, seq_len, _ = price_input.shape
        
        # Encode price
        price_encoded = self.price_input_proj(price_input)
        price_encoded = price_encoded + self.pos_encoding[:, :seq_len, :]
        
        # LSTM
        price_lstm_out, _ = self.price_lstm(price_encoded)
        
        # Process text if available
        text_attn_weights = None
        if text_input is not None and self.text_encoder is not None:
            # Encode text
            text_encoded = self.text_encoder(text_input)
            
            # Cross-modal attention
            if self.text_price_attention is not None:
                fused, text_attn_weights = self.text_price_attention(
                    price_lstm_out,
                    text_encoded,
                    return_weights=return_attention
                )
            else:
                fused = price_lstm_out
            
            # Late fusion
            if self.config.fusion_method == "late":
                # Aggregate text features
                text_pooled = text_encoded.mean(dim=1, keepdim=True)
                text_pooled = text_pooled.expand(-1, seq_len, -1)
                fused = torch.cat([fused, text_pooled], dim=-1)
                fused = self.fusion(fused)
        else:
            fused = price_lstm_out
        
        # Self-attention
        attn_out, temporal_attn_weights = self.self_attention(
            fused, fused, fused
        )
        
        # Post-attention
        output = self.post_attention(attn_out + fused)
        
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
            "temporal_attention": temporal_attn_weights
        }
        
        if return_attention and text_attn_weights is not None:
            result["text_attention"] = text_attn_weights
        
        return result
    
    def predict(
        self,
        price_input: torch.Tensor,
        text_input: torch.Tensor = None,
        quantile: float = 0.5
    ) -> torch.Tensor:
        """
        Get point predictions.
        
        Args:
            price_input: Price features
            text_input: Text embeddings
            quantile: Quantile to return (0.5 for median)
            
        Returns:
            Point predictions
        """
        with torch.no_grad():
            output = self.forward(price_input, text_input)
            predictions = output["predictions"]
            quantile_idx = self.config.quantiles.index(quantile)
            return predictions[..., quantile_idx]
    
    def get_text_importance(
        self,
        price_input: torch.Tensor,
        text_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Get importance of text features via attention.
        
        Args:
            price_input: Price features
            text_input: Text embeddings
            
        Returns:
            Text attention weights [batch, price_seq, text_seq]
        """
        with torch.no_grad():
            output = self.forward(price_input, text_input, return_attention=True)
            return output.get("text_attention")


class SentimentEnhancedTFT(nn.Module):
    """
    Simplified TFT with sentiment features as static covariates.
    
    Uses sentiment scores directly as numeric features alongside price.
    Lighter weight alternative to full Multimodal TFT.
    """
    
    def __init__(
        self,
        price_features: int = 1,
        sentiment_features: int = 4,  # pos, neg, neu, score
        hidden_size: int = 64,
        prediction_horizon: int = 12,
        context_length: int = 60,
        dropout: float = 0.1
    ):
        super().__init__()
        
        total_features = price_features + sentiment_features
        
        # Feature projection
        self.input_proj = nn.Linear(total_features, hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output
        self.output_proj = nn.Linear(hidden_size, prediction_horizon)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        price_input: torch.Tensor,
        sentiment_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            price_input: [batch, seq_len, price_features]
            sentiment_input: [batch, seq_len, sentiment_features]
            
        Returns:
            Predictions dict
        """
        # Concatenate inputs
        x = torch.cat([price_input, sentiment_input], dim=-1)
        
        # Project
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        
        # Residual
        output = lstm_out + attn_out
        
        # Use last timestep
        output = output[:, -1, :]
        
        # Project to predictions
        predictions = self.output_proj(output)
        
        return {
            "predictions": predictions,
            "attention_weights": attn_weights
        }
    
    def predict(
        self,
        price_input: torch.Tensor,
        sentiment_input: torch.Tensor
    ) -> torch.Tensor:
        """Get point predictions."""
        with torch.no_grad():
            return self.forward(price_input, sentiment_input)["predictions"]
