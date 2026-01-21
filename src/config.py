# Commodities Price Forecasting
# State-of-the-art ML/DL for non-ferrous metals price prediction

"""
Project configuration and constants.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Data parameters
    context_length: int = 60  # Lookback window (months)
    prediction_horizon: int = 12  # Max forecast horizon
    lead_times: List[int] = None  # Specific lead times to evaluate
    
    # Training parameters
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    
    # Model-specific
    hidden_size: int = 64
    attention_heads: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.lead_times is None:
            self.lead_times = [1, 2, 3, 4, 5, 6, 12]


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_obv: bool = True
    
    # Rolling statistics
    rolling_windows: List[int] = None
    
    # Lag features
    max_lag_months: int = 12
    use_auto_lag_selection: bool = True
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12, 24]


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()

# Commodity symbols
COMMODITIES = {
    "aluminum": "AL",
    "copper": "CU",
    "zinc": "ZN"
}
