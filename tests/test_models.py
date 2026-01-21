"""
Unit tests for Models
"""

import unittest
import torch
import numpy as np
from src.models.tft import TFTModel, TFTConfig
from src.models.nbeats import NBEATSModel, NBEATSConfig
from src.models.multimodal_tft import MultimodalTFT, MultimodalTFTConfig

class TestModels(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 30
        self.input_size = 5
        self.prediction_horizon = 5
        
        # Create dummy input [batch, seq_len, features]
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_size)

    def test_tft_forward(self):
        config = TFTConfig(
            input_size=self.input_size,
            hidden_size=16,
            context_length=self.seq_len,
            prediction_horizon=self.prediction_horizon
        )
        model = TFTModel(config)
        output = model(self.x)
        
        # Check output shape [batch, horizon, quantiles]
        # Default quantiles is 3
        expected_shape = (self.batch_size, self.prediction_horizon, 3)
        
        if isinstance(output, dict):
            pred = output['predictions']
        else:
            pred = output
            
        self.assertEqual(pred.shape, expected_shape)

    def test_nbeats_forward(self):
        config = NBEATSConfig(
            input_size=self.seq_len, # N-BEATS takes univariate history usually, but depends on impl
            output_size=self.prediction_horizon,
            hidden_size=16
        )
        # N-BEATS usually expects [batch, input_size] for univariate
        x_univariate = self.x[:, :, 0] 
        model = NBEATSModel(config)
        output = model(x_univariate)
        
        # Check output shape: expect {forecast: [batch, horizon]} or similar
        if isinstance(output, dict):
            pred = output['forecast']
        else:
            pred = output
            
        self.assertEqual(pred.shape, (self.batch_size, self.prediction_horizon))

    def test_multimodal_tft_forward(self):
        text_dim = 32
        text_input = torch.randn(self.batch_size, self.seq_len, text_dim)
        
        config = MultimodalTFTConfig(
            price_input_size=self.input_size,
            text_embedding_dim=text_dim,
            hidden_size=16,
            context_length=self.seq_len,
            prediction_horizon=self.prediction_horizon
        )
        model = MultimodalTFT(config)
        output = model(self.x, text_input)
        
        pred = output['predictions']
        self.assertEqual(pred.shape, (self.batch_size, self.prediction_horizon, 3))

if __name__ == '__main__':
    unittest.main()
