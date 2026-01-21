"""
Feature Engineering Module

Advanced feature engineering pipeline for commodity price forecasting.
Implements rolling statistics, lag selection, and time-based features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from statsmodels.tsa.stattools import pacf
from scipy import stats


class FeatureEngineer:
    """
    Feature engineering pipeline for time series data.
    
    Implements:
    - Rolling window statistics
    - Optimal lag selection using PACF
    - Time-based features
    - Return and volatility features
    """
    
    def __init__(
        self,
        rolling_windows: List[int] = None,
        max_lag: int = 12,
        pacf_significance_level: float = 0.05
    ):
        """
        Initialize Feature Engineer.
        
        Args:
            rolling_windows: List of rolling window sizes (default [3, 6, 12, 24])
            max_lag: Maximum lag to consider for feature selection (default 12)
            pacf_significance_level: Significance level for PACF-based lag selection
        """
        self.rolling_windows = rolling_windows or [3, 6, 12, 24]
        self.max_lag = max_lag
        self.pacf_significance_level = pacf_significance_level
    
    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Add rolling window statistics for specified columns.
        
        Adds: mean, std, min, max, skew, kurtosis for each window.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute statistics for
            windows: Window sizes (uses self.rolling_windows if None)
            
        Returns:
            DataFrame with rolling statistics added
        """
        result = df.copy()
        windows = windows or self.rolling_windows
        
        for col in columns:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            for window in windows:
                # Basic statistics
                result[f"{col}_roll_mean_{window}"] = series.rolling(window).mean()
                result[f"{col}_roll_std_{window}"] = series.rolling(window).std()
                result[f"{col}_roll_min_{window}"] = series.rolling(window).min()
                result[f"{col}_roll_max_{window}"] = series.rolling(window).max()
                
                # Advanced statistics
                result[f"{col}_roll_skew_{window}"] = series.rolling(window).skew()
                result[f"{col}_roll_kurt_{window}"] = series.rolling(window).kurt()
                
                # Range as percentage of mean
                result[f"{col}_roll_range_{window}"] = (
                    (result[f"{col}_roll_max_{window}"] - result[f"{col}_roll_min_{window}"]) /
                    result[f"{col}_roll_mean_{window}"]
                )
        
        return result
    
    def select_optimal_lags(
        self,
        series: pd.Series,
        max_lag: int = None,
        significance_level: float = None
    ) -> List[int]:
        """
        Select optimal lags using Partial Autocorrelation Function.
        
        Lags with PACF values significantly different from zero are selected.
        
        Args:
            series: Time series to analyze
            max_lag: Maximum lag to consider
            significance_level: Significance level for selection
            
        Returns:
            List of significant lag values
        """
        max_lag = max_lag or self.max_lag
        significance_level = significance_level or self.pacf_significance_level
        
        # Compute PACF
        pacf_values = pacf(series.dropna(), nlags=max_lag)
        
        # Compute confidence interval (approximate)
        n = len(series.dropna())
        conf_int = stats.norm.ppf(1 - significance_level / 2) / np.sqrt(n)
        
        # Select significant lags (excluding lag 0)
        significant_lags = [
            i for i in range(1, len(pacf_values))
            if abs(pacf_values[i]) > conf_int
        ]
        
        return significant_lags
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = None,
        auto_select: bool = True
    ) -> pd.DataFrame:
        """
        Add lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: Specific lags to use (if auto_select=False)
            auto_select: Whether to automatically select optimal lags
            
        Returns:
            DataFrame with lag features added
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col]
            
            if auto_select:
                selected_lags = self.select_optimal_lags(series)
                # Always include at least lag 1
                if 1 not in selected_lags:
                    selected_lags = [1] + selected_lags
            else:
                selected_lags = lags or list(range(1, self.max_lag + 1))
            
            for lag in selected_lags:
                result[f"{col}_lag_{lag}"] = series.shift(lag)
        
        return result
    
    def add_return_features(
        self,
        df: pd.DataFrame,
        price_col: str,
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Add return-based features.
        
        Adds: simple returns, log returns, and return volatility.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            periods: Return periods to compute (default [1, 3, 6, 12])
            
        Returns:
            DataFrame with return features added
        """
        result = df.copy()
        periods = periods or [1, 3, 6, 12]
        prices = df[price_col]
        
        for period in periods:
            # Simple return
            result[f"return_{period}"] = prices.pct_change(period)
            
            # Log return
            result[f"log_return_{period}"] = np.log(prices / prices.shift(period))
            
            # Return volatility (rolling std of returns)
            if period > 1:
                result[f"return_vol_{period}"] = (
                    result[f"return_1"].rolling(period).std() if "return_1" in result.columns
                    else prices.pct_change(1).rolling(period).std()
                )
        
        return result
    
    def add_time_features(
        self,
        df: pd.DataFrame,
        date_col: str = None
    ) -> pd.DataFrame:
        """
        Add time-based features from datetime index or column.
        
        Adds: month, quarter, year, and cyclical encodings.
        
        Args:
            df: Input DataFrame with datetime index or date column
            date_col: Date column name (uses index if None)
            
        Returns:
            DataFrame with time features added
        """
        result = df.copy()
        
        if date_col:
            dates = pd.to_datetime(df[date_col])
            # Use .dt accessor if it's a Series
            if isinstance(dates, pd.Series):
                dates = dates.dt
        else:
            dates = pd.to_datetime(df.index)
        
        # Basic time features
        result["month"] = dates.month
        result["quarter"] = dates.quarter
        result["year"] = dates.year
        result["day_of_year"] = dates.dayofyear
        result["day_of_week"] = dates.dayofweek
        
        # Cyclical encoding for month (sin/cos transform)
        result["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
        result["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
        
        # Cyclical encoding for quarter
        result["quarter_sin"] = np.sin(2 * np.pi * dates.quarter / 4)
        result["quarter_cos"] = np.cos(2 * np.pi * dates.quarter / 4)
        
        return result
    
    def add_target_features(
        self,
        df: pd.DataFrame,
        price_col: str,
        lead_times: List[int]
    ) -> pd.DataFrame:
        """
        Add target variables for different lead times.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            lead_times: List of lead times (forecast horizons)
            
        Returns:
            DataFrame with target columns added
        """
        result = df.copy()
        prices = df[price_col]
        
        for lead_time in lead_times:
            # Future price
            result[f"target_{lead_time}"] = prices.shift(-lead_time)
            
            # Future return
            result[f"target_return_{lead_time}"] = (
                prices.shift(-lead_time) - prices
            ) / prices
        
        return result
    
    def create_full_feature_set(
        self,
        df: pd.DataFrame,
        price_col: str,
        lead_times: List[int],
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        volume_col: Optional[str] = None,
        additional_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create complete feature set combining all feature types.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            lead_times: Forecast horizons
            high_col: High price column (optional)
            low_col: Low price column (optional)
            volume_col: Volume column (optional)
            additional_cols: Other columns to add features for
            
        Returns:
            DataFrame with full feature set
        """
        from .technical import TechnicalIndicators
        
        result = df.copy()
        
        # Add technical indicators
        result = TechnicalIndicators.add_all_indicators(
            result,
            price_col=price_col,
            high_col=high_col,
            low_col=low_col,
            volume_col=volume_col
        )
        
        # Add rolling statistics for price
        result = self.add_rolling_statistics(result, [price_col])
        
        # Add lag features
        cols_to_lag = [price_col] + (additional_cols or [])
        result = self.add_lag_features(result, cols_to_lag)
        
        # Add return features
        result = self.add_return_features(result, price_col)
        
        # Add time features
        result = self.add_time_features(result)
        
        # Add targets
        result = self.add_target_features(result, price_col, lead_times)
        
        return result
