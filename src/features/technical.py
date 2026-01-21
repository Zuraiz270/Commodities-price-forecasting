"""
Technical Indicators Module

Implementation of state-of-the-art technical indicators for commodity price forecasting.
Based on latest research (2024-2025) on feature engineering for financial time series.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


class TechnicalIndicators:
    """
    Collection of technical indicators for commodity price analysis.
    
    Implements:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    - OBV (On-Balance Volume)
    """
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        RSI measures momentum and identifies overbought/oversold conditions.
        Values above 70 indicate overbought, below 30 indicate oversold.
        
        Args:
            prices: Price series (typically close prices)
            period: Lookback period (default 14)
            
        Returns:
            RSI values as pandas Series
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.rename(f"RSI_{period}")
    
    @staticmethod
    def macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD is a trend-following momentum indicator showing the relationship
        between two moving averages of a security's price.
        
        Args:
            prices: Price series
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with MACD line, signal line, and histogram
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": histogram
        })
    
    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands are volatility bands placed above and below a moving average.
        Width of bands adjusts based on volatility.
        
        Args:
            prices: Price series
            period: Moving average period (default 20)
            std_dev: Number of standard deviations for bands (default 2)
            
        Returns:
            DataFrame with upper band, middle band (SMA), lower band, and bandwidth
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = (upper - lower) / middle
        
        # %B indicator: where price is relative to bands
        percent_b = (prices - lower) / (upper - lower)
        
        return pd.DataFrame({
            "BB_Upper": upper,
            "BB_Middle": middle,
            "BB_Lower": lower,
            "BB_Width": bandwidth,
            "BB_PercentB": percent_b
        })
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        ATR measures market volatility by decomposing the entire range
        of an asset price for that period.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)
            
        Returns:
            ATR values as pandas Series
        """
        high_low = high - low
        high_close_prev = abs(high - close.shift(1))
        low_close_prev = abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr.rename(f"ATR_{period}")
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV uses volume flow to predict changes in stock price.
        Rising OBV reflects positive volume pressure that can lead to higher prices.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV values as pandas Series
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        
        obv = (direction * volume).cumsum()
        
        return obv.rename("OBV")
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate price momentum.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Momentum values as pandas Series
        """
        return (prices - prices.shift(period)).rename(f"Momentum_{period}")
    
    @staticmethod
    def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC).
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            ROC values as pandas Series (percentage)
        """
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc.rename(f"ROC_{period}")
    
    @staticmethod
    def stochastic_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default 14)
            d_period: %D smoothing period (default 3)
            
        Returns:
            DataFrame with %K and %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            "Stoch_K": stoch_k,
            "Stoch_D": stoch_d
        })
    
    @classmethod
    def add_all_indicators(
        cls,
        df: pd.DataFrame,
        price_col: str = "Close",
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        volume_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add all available technical indicators to a DataFrame.
        
        Args:
            df: Input DataFrame with price data
            price_col: Column name for price (default "Close")
            high_col: Column name for high price (optional)
            low_col: Column name for low price (optional)
            volume_col: Column name for volume (optional)
            
        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()
        prices = df[price_col]
        
        # Always add these
        result[f"RSI_14"] = cls.rsi(prices, 14)
        result[f"RSI_7"] = cls.rsi(prices, 7)
        
        macd_df = cls.macd(prices)
        result = pd.concat([result, macd_df], axis=1)
        
        bb_df = cls.bollinger_bands(prices)
        result = pd.concat([result, bb_df], axis=1)
        
        result["Momentum_10"] = cls.momentum(prices, 10)
        result["ROC_10"] = cls.rate_of_change(prices, 10)
        
        # Add if high/low available
        if high_col and low_col:
            high, low = df[high_col], df[low_col]
            result["ATR_14"] = cls.atr(high, low, prices, 14)
            
            stoch_df = cls.stochastic_oscillator(high, low, prices)
            result = pd.concat([result, stoch_df], axis=1)
        
        # Add if volume available
        if volume_col:
            result["OBV"] = cls.obv(prices, df[volume_col])
        
        return result
