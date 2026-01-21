"""
End-to-End Forecasting Example

This script demonstrates the full workflow:
1. Data Loading (Synthetic)
2. Feature Engineering
3. Training (TFT)
4. Evaluation
5. Interpretation (SHAP)
"""

import pandas as pd
import numpy as np
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.technical import TechnicalIndicators
from src.features.engineering import FeatureEngineer
from src.models.tft import TFTModel, TFTConfig
from src.training.trainer import Trainer
from src.utils.data_loader import DataLoader as CommodityDataLoader
from src.interpretability.shap_analysis import SHAPAnalyzer

def generate_synthetic_data(n_days=500):
    """Generate synthetic commodity price data."""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Random walk with trend and seasonality
    t = np.arange(n_days)
    trend = t * 0.1
    seasonality = 10 * np.sin(2 * np.pi * t / 365)
    noise = np.random.randn(n_days) * 2
    price = 1000 + trend + seasonality + noise
    
    # Create Open, High, Low, Close, Volume
    close = price
    open_ = close + np.random.randn(n_days) * 5
    high = np.maximum(open_, close) + np.random.rand(n_days) * 10
    low = np.minimum(open_, close) - np.random.rand(n_days) * 10
    volume = np.random.randint(1000, 10000, n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df

def main():
    print("🚀 Starting End-to-End Forecasting Demo")
    
    # 1. Data Loading
    print("\n📦 Generating Synthetic Data...")
    df = generate_synthetic_data()
    print(f"Data shape: {df.shape}")
    
    # 2. Feature Engineering
    print("\n🛠️ Feature Engineering...")
    # Add technical indicators
    ti = TechnicalIndicators()
    df['rsi'] = ti.rsi(df['close'])
    macd_df = ti.macd(df['close'])
    df['macd'] = macd_df['MACD']
    
    # Add rolling stats
    fe = FeatureEngineer()
    df = fe.add_rolling_statistics(df, columns=['close'], windows=[7, 30])
    df = fe.add_time_features(df, 'date')
    
    # Drop NaN
    df = df.dropna()
    print(f"Features created: {df.columns.tolist()}")
    
    # 3. Preprocessing
    print("\n🧹 Preprocessing...")
    loader = CommodityDataLoader()
    # Normalize features
    target_col = 'close'
    feature_cols = ['close', 'open', 'high', 'low', 'volume', 'rsi', 'macd']
    
    # Quick normalization
    data = df[feature_cols].values
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        data,
        context_length=30,
        prediction_horizon=5,
        batch_size=32
    )
    
    # 4. Model Training
    print("\n🏋️ Training TFT Model...")
    config = TFTConfig(
        input_size=len(feature_cols),
        hidden_size=32,
        context_length=30,
        prediction_horizon=5,
        attention_heads=2
    )
    model = TFTModel(config)
    
    trainer = Trainer(model)
    # Train for just a few epochs for demo
    trainer.config.max_epochs = 2
    history = trainer.fit(train_loader, val_loader)
    
    # 5. Evaluation
    print("\n📊 Evaluation...")
    metrics = trainer.evaluate(test_loader)
    print(f"Test MAE: {metrics['MAE']:.4f}")
    
    # 6. Interpretation
    print("\n🔍 SHAP Interpretation...")
    try:
        # Create background dataset for SHAP
        background = next(iter(train_loader))[0].numpy()[:10] 
        
        # Initialize analyzer
        analyzer = SHAPAnalyzer(model)
        analyzer.fit(background)
        
        # Explain one batch
        x_test = next(iter(test_loader))[0].numpy()[:5]
        shap_vals = analyzer.explain(x_test)
        
        # Get importance
        importance = analyzer.feature_importance(x_test)
        print("Top Feature Importance:")
        # Map index to feature names
        importance['feature_name'] = [feature_cols[int(f.split('_')[1])] if 'Feature_' in f else f for f in importance['feature']]
        print(importance.head())
    except ImportError:
        print("⚠️ SHAP not installed. Skipping interpretation step.")
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
    
    print("\n✅ Demo Completed Successfully!")

if __name__ == "__main__":
    main()
