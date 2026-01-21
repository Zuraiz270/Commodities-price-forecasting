"""
NLP Forecasting Demo

Demonstrates how to use the NLP modules for commodity price forecasting.
"""

import pandas as pd
import numpy as np
from src.nlp.sentiment import FinBERTSentiment, SentimentAggregator
from src.nlp.news_fetcher import NewsFetcher
from src.nlp.event_extraction import EventExtractor
from src.models.multimodal_tft import MultimodalTFT, MultimodalTFTConfig

def main():
    print("🚀 Starting NLP Forecasting Demo")
    
    # 1. Fetch News
    print("\n📰 Fetching News...")
    fetcher = NewsFetcher()
    # Using sample data since no API key provided
    news_df = fetcher.create_sample_news(
        commodity="aluminum",
        n_samples=20
    )
    print(f"Fetched {len(news_df)} articles")
    print(news_df[["published_at", "headline"]].head(3))
    
    # 2. Analyze Sentiment (FinBERT)
    print("\n🧠 Analyzing Sentiment (FinBERT)...")
    # Using CPU for demo, normalized batch size
    analyzer = FinBERTSentiment(device="cpu", batch_size=4)
    aggregator = SentimentAggregator(analyzer)
    
    daily_sentiment = aggregator.aggregate_daily(news_df)
    print("\nDaily Sentiment Features:")
    print(daily_sentiment.head(3))
    
    # 3. Extract Events (LLM)
    print("\n🔍 Extracting Events...")
    extractor = EventExtractor(use_local=True) # Rule-based fallback
    events = extractor.extract_batch(news_df["headline"].tolist())
    
    event_features = extractor.create_event_features(events)
    print(f"\nExtracted {len(events)} events")
    print("Event Features sample:")
    print(event_features.head(3))
    
    # 4. Multimodal Model Setup
    print("\n🤖 Setting up Multimodal TFT...")
    config = MultimodalTFTConfig(
        price_input_size=5,  # price, high, low, vol, open
        context_length=10,
        prediction_horizon=5,
        use_text_attention=True
    )
    
    model = MultimodalTFT(config)
    print(model)
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
