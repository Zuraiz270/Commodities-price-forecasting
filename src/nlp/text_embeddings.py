"""
Text Embeddings Module

Generate text embeddings from news for multimodal forecasting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import torch
import torch.nn as nn


class TextEmbedder:
    """
    Generate embeddings from news text for price forecasting.
    
    Supports multiple embedding methods:
    - Sentence Transformers (lightweight, fast)
    - OpenAI embeddings (high quality, requires API)
    - Custom trained embeddings
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto"
    ):
        """
        Initialize text embedder.
        
        Args:
            model_name: Sentence Transformer model name
            device: Device to use
        """
        self.model_name = model_name
        self.model = None
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def load_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Embeddings array [n_texts, embedding_dim]
        """
        if self.model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings
    
    def embed_with_pooling(
        self,
        texts: List[str],
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Embed multiple texts and pool to single vector.
        
        Useful for aggregating multiple news articles per day.
        
        Args:
            texts: List of texts
            pooling: Pooling method ('mean', 'max', 'first')
            
        Returns:
            Single embedding vector
        """
        embeddings = self.embed(texts)
        
        if pooling == "mean":
            return embeddings.mean(axis=0)
        elif pooling == "max":
            return embeddings.max(axis=0)
        elif pooling == "first":
            return embeddings[0]
        else:
            return embeddings.mean(axis=0)
    
    def create_daily_embeddings(
        self,
        news_df: pd.DataFrame,
        text_col: str = "headline",
        date_col: str = "date",
        pooling: str = "mean"
    ) -> pd.DataFrame:
        """
        Create daily text embeddings from news.
        
        Args:
            news_df: DataFrame with news articles
            text_col: Column containing text
            date_col: Column containing dates
            pooling: Pooling method for multiple articles
            
        Returns:
            DataFrame with date and embedding columns
        """
        news_df = news_df.copy()
        news_df[date_col] = pd.to_datetime(news_df[date_col]).dt.date
        
        daily_embeddings = []
        
        for date, group in news_df.groupby(date_col):
            texts = group[text_col].tolist()
            embedding = self.embed_with_pooling(texts, pooling)
            
            daily_embeddings.append({
                date_col: date,
                "embedding": embedding,
                "n_articles": len(texts)
            })
        
        df = pd.DataFrame(daily_embeddings)
        
        # Expand embedding to columns
        embedding_cols = pd.DataFrame(
            df["embedding"].tolist(),
            columns=[f"text_emb_{i}" for i in range(len(df["embedding"].iloc[0]))]
        )
        
        result = pd.concat([
            df[[date_col, "n_articles"]],
            embedding_cols
        ], axis=1)
        
        return result


class FinancialTextEncoder(nn.Module):
    """
    Trainable text encoder for financial news.
    
    Can be jointly trained with price forecasting model.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Token embedding dimension
            hidden_dim: LSTM hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode text to fixed-size embedding.
        
        Args:
            x: Token IDs [batch, seq_len]
            
        Returns:
            Embeddings [batch, output_dim]
        """
        # Embed tokens
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use final hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # Project to output dimension
        output = self.fc(hidden)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between text and price features.
    
    Allows the model to attend to relevant news when making predictions.
    """
    
    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            price_dim: Dimension of price features
            text_dim: Dimension of text embeddings
            hidden_dim: Attention hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        # Project to common dimension
        self.price_proj = nn.Linear(price_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(
        self,
        price_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal attention.
        
        Args:
            price_features: [batch, seq_len, price_dim]
            text_features: [batch, n_texts, text_dim]
            
        Returns:
            Fused features [batch, seq_len, hidden_dim]
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
        
        # Concatenate and project
        fused = torch.cat([price_proj, attn_out], dim=-1)
        output = self.output_proj(fused)
        
        return output
