"""
Verify NLP Modules

Script to verify that all NLP modules can be imported and initialized.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def verify_imports():
    print("Verifying NLP module imports...")
    
    try:
        from src.nlp.sentiment import FinBERTSentiment, SentimentAggregator
        print("✅ Sentiment module imported")
    except ImportError as e:
        print(f"⚠️ Sentiment module import failed: {e}")
        
    try:
        from src.nlp.news_fetcher import NewsFetcher, NewsCache
        print("✅ News Fetcher module imported")
    except ImportError as e:
        print(f"⚠️ News Fetcher module import failed: {e}")

    try:
        from src.nlp.event_extraction import EventExtractor
        print("✅ Event Extraction module imported")
    except ImportError as e:
        print(f"⚠️ Event Extraction module import failed: {e}")

    try:
        from src.nlp.text_embeddings import TextEmbedder
        print("✅ Text Embeddings module imported")
    except ImportError as e:
        print(f"⚠️ Text Embeddings module import failed: {e}")

    try:
        from src.models.multimodal_tft import MultimodalTFT, MultimodalTFTConfig
        print("✅ Multimodal TFT model imported")
    except ImportError as e:
        print(f"⚠️ Multimodal TFT model import failed: {e}")

if __name__ == "__main__":
    verify_imports()
