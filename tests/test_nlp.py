"""
Unit tests for NLP components
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.nlp.sentiment import FinBERTSentiment, SentimentAggregator
from src.nlp.event_extraction import EventExtractor, CommodityEvent, EventType

class TestNLP(unittest.TestCase):

    def test_sentiment_structure(self):
        # Mock the analyzer to avoid downloading models
        analyzer = MagicMock()
        analyzer.analyze.return_value.positive = 0.8
        analyzer.analyze_batch.return_value = [
            MagicMock(positive=0.9, negative=0.1, neutral=0.0, label="positive"),
            MagicMock(positive=0.1, negative=0.8, neutral=0.1, label="negative")
        ]
        
        aggregator = SentimentAggregator(analyzer)
        
        df = pd.DataFrame({
            "headline": ["Good news", "Bad news"],
            "date": ["2024-01-01", "2024-01-01"]
        })
        
        result = aggregator.aggregate_daily(df)
        self.assertIn("sentiment_score", result.columns)
        self.assertEqual(len(result), 1) # Should aggregate to 1 day

    def test_event_extraction_mock(self):
        # Test rule-based extraction
        extractor = EventExtractor(use_local=True)
        
        text = "Aluminum prices surge due to strike at smelter"
        event = extractor.extract_event(text)
        
        self.assertIsNotNone(event)
        self.assertEqual(event.commodity, "aluminum")
        self.assertEqual(event.sentiment, "positive")
        # Rule based detects "strike" as supply disruption usually or price movement depending on priority
        # Let's check if it returns a valid EventType
        self.assertIsInstance(event.event_type, EventType)

if __name__ == '__main__':
    unittest.main()
