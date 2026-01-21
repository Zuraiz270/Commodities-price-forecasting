"""
Financial Sentiment Analysis Module

Uses FinBERT (ProsusAI/finbert) for domain-specific financial sentiment analysis
of commodity market news and reports.

Based on research: "Analyzing financial news sentiment" for price prediction
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    text: str
    positive: float
    negative: float
    neutral: float
    label: str
    confidence: float


class FinBERTSentiment:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    FinBERT is trained on financial communications and outperforms
    generic sentiment models (VADER, TextBlob) on financial text.
    
    Reference: Huang et al. "FinBERT: A Pre-trained Financial Language 
    Representation Model for Financial Text Mining"
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('auto', 'cpu', 'cuda')
            batch_size: Batch size for inference
            max_length: Maximum token length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        # Set device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            print(f"Loading FinBERT from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"FinBERT loaded on {self.device}")
            
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with scores and label
        """
        if self.model is None:
            self.load_model()
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        probs = probs.cpu().numpy()[0]
        
        # FinBERT outputs: [positive, negative, neutral]
        positive, negative, neutral = probs[0], probs[1], probs[2]
        
        # Determine label
        label_idx = np.argmax(probs)
        labels = ["positive", "negative", "neutral"]
        label = labels[label_idx]
        confidence = float(probs[label_idx])
        
        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            positive=float(positive),
            negative=float(negative),
            neutral=float(neutral),
            label=label,
            confidence=confidence
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        if self.model is None:
            self.load_model()
        
        import torch
        
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            probs = probs.cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                p = probs[j]
                label_idx = np.argmax(p)
                labels = ["positive", "negative", "neutral"]
                
                results.append(SentimentResult(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    positive=float(p[0]),
                    negative=float(p[1]),
                    neutral=float(p[2]),
                    label=labels[label_idx],
                    confidence=float(p[label_idx])
                ))
        
        return results
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get a single sentiment score (-1 to 1).
        
        -1 = very negative, 0 = neutral, +1 = very positive
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score
        """
        result = self.analyze(text)
        return result.positive - result.negative


class SentimentAggregator:
    """
    Aggregate sentiment scores to daily/weekly features for time series.
    
    Provides multiple aggregation strategies for combining multiple
    news articles into a single feature per time period.
    """
    
    def __init__(
        self,
        sentiment_analyzer: FinBERTSentiment = None,
        aggregation: str = "mean"
    ):
        """
        Initialize aggregator.
        
        Args:
            sentiment_analyzer: FinBERT analyzer instance
            aggregation: Aggregation method ('mean', 'weighted', 'max')
        """
        self.analyzer = sentiment_analyzer or FinBERTSentiment()
        self.aggregation = aggregation
    
    def aggregate_daily(
        self,
        news_df: pd.DataFrame,
        text_col: str = "headline",
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment to daily features.
        
        Args:
            news_df: DataFrame with news articles
            text_col: Column containing text
            date_col: Column containing dates
            
        Returns:
            DataFrame with daily sentiment features
        """
        news_df = news_df.copy()
        news_df[date_col] = pd.to_datetime(news_df[date_col]).dt.date
        
        # Analyze all texts
        texts = news_df[text_col].tolist()
        results = self.analyzer.analyze_batch(texts)
        
        # Add sentiment columns
        news_df["sentiment_positive"] = [r.positive for r in results]
        news_df["sentiment_negative"] = [r.negative for r in results]
        news_df["sentiment_neutral"] = [r.neutral for r in results]
        news_df["sentiment_score"] = [r.positive - r.negative for r in results]
        news_df["sentiment_label"] = [r.label for r in results]
        
        # Aggregate by date
        if self.aggregation == "mean":
            daily = news_df.groupby(date_col).agg({
                "sentiment_positive": "mean",
                "sentiment_negative": "mean",
                "sentiment_neutral": "mean",
                "sentiment_score": "mean"
            }).reset_index()
        elif self.aggregation == "max":
            # Use most extreme sentiment
            daily = news_df.groupby(date_col).agg({
                "sentiment_positive": "max",
                "sentiment_negative": "max",
                "sentiment_neutral": "min",
                "sentiment_score": lambda x: x[np.abs(x).argmax()]
            }).reset_index()
        else:
            daily = news_df.groupby(date_col).agg({
                "sentiment_positive": "mean",
                "sentiment_negative": "mean",
                "sentiment_neutral": "mean",
                "sentiment_score": "mean"
            }).reset_index()
        
        # Add article count
        counts = news_df.groupby(date_col).size().reset_index(name="news_count")
        daily = daily.merge(counts, on=date_col)
        
        return daily
    
    def create_sentiment_features(
        self,
        news_df: pd.DataFrame,
        price_df: pd.DataFrame,
        text_col: str = "headline",
        date_col: str = "date",
        price_date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Create sentiment features aligned with price data.
        
        Args:
            news_df: DataFrame with news articles
            price_df: DataFrame with price data
            text_col: News text column
            date_col: News date column
            price_date_col: Price date column
            
        Returns:
            Price DataFrame with sentiment features added
        """
        # Get daily sentiment
        daily_sentiment = self.aggregate_daily(news_df, text_col, date_col)
        
        # Ensure dates are comparable
        price_df = price_df.copy()
        price_df[price_date_col] = pd.to_datetime(price_df[price_date_col]).dt.date
        daily_sentiment[date_col] = pd.to_datetime(daily_sentiment[date_col]).dt.date
        
        # Merge with price data
        result = price_df.merge(
            daily_sentiment,
            left_on=price_date_col,
            right_on=date_col,
            how="left"
        )
        
        # Fill missing sentiment (weekends/holidays) with last known
        sentiment_cols = [
            "sentiment_positive", "sentiment_negative",
            "sentiment_neutral", "sentiment_score", "news_count"
        ]
        for col in sentiment_cols:
            if col in result.columns:
                result[col] = result[col].fillna(method="ffill").fillna(0)
        
        return result
    
    def compute_sentiment_momentum(
        self,
        sentiment_series: pd.Series,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Compute sentiment momentum features.
        
        Momentum: change in sentiment over time, useful for
        detecting sentiment shifts before price moves.
        
        Args:
            sentiment_series: Daily sentiment scores
            windows: Lookback windows for momentum
            
        Returns:
            DataFrame with momentum features
        """
        windows = windows or [3, 7, 14, 30]
        
        features = pd.DataFrame(index=sentiment_series.index)
        
        for window in windows:
            # Rolling mean
            features[f"sentiment_ma_{window}"] = (
                sentiment_series.rolling(window).mean()
            )
            
            # Momentum (change from N days ago)
            features[f"sentiment_momentum_{window}"] = (
                sentiment_series - sentiment_series.shift(window)
            )
            
            # Rolling std (sentiment volatility)
            features[f"sentiment_vol_{window}"] = (
                sentiment_series.rolling(window).std()
            )
        
        return features


class VADERSentiment:
    """
    VADER sentiment as lightweight alternative to FinBERT.
    
    Faster but less accurate for financial text.
    Use for quick prototyping or when GPU unavailable.
    """
    
    def __init__(self):
        """Initialize VADER."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError(
                "vaderSentiment not installed. Install with: pip install vaderSentiment"
            )
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze text with VADER."""
        scores = self.analyzer.polarity_scores(text)
        
        # Map VADER output
        positive = scores["pos"]
        negative = scores["neg"]
        neutral = scores["neu"]
        compound = scores["compound"]
        
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            positive=positive,
            negative=negative,
            neutral=neutral,
            label=label,
            confidence=abs(compound)
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def get_sentiment_score(self, text: str) -> float:
        """Get compound score (-1 to 1)."""
        return self.analyzer.polarity_scores(text)["compound"]
