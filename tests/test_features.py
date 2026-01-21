"""
Unit tests for Feature Engineering
"""

import unittest
import pandas as pd
import numpy as np
from src.features.technical import TechnicalIndicators
from src.features.engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        # Create dummy price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'date': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'volume': np.random.randint(100, 1000, 100)
        })
        self.ti = TechnicalIndicators()
        self.fe = FeatureEngineer()

    def test_rsi(self):
        rsi = self.ti.rsi(self.df['close'], period=14)
        self.assertEqual(len(rsi), 100)
        self.assertTrue(rsi.max() <= 100)
        self.assertTrue(rsi.min() >= 0)

    def test_macd(self):
        macd_df = self.ti.macd(self.df['close'])
        self.assertIn("MACD", macd_df.columns)
        self.assertIn("MACD_Signal", macd_df.columns)

    def test_bollinger_bands(self):
        bb_df = self.ti.bollinger_bands(self.df['close'])
        self.assertIn("BB_Upper", bb_df.columns)
        # Drop NaNs created by rolling window
        bb_df = bb_df.dropna()
        self.assertTrue(all(bb_df["BB_Upper"] >= bb_df["BB_Middle"]))

    def test_rolling_stats(self):
        stats = self.fe.add_rolling_statistics(self.df, columns=['close'], windows=[5])
        self.assertIn('close_roll_mean_5', stats.columns)
        self.assertIn('close_roll_std_5', stats.columns)

    def test_time_features(self):
        time_feats = self.fe.add_time_features(self.df, 'date')
        self.assertIn('month_sin', time_feats.columns)
        self.assertIn('day_of_week', time_feats.columns)

if __name__ == '__main__':
    unittest.main()
