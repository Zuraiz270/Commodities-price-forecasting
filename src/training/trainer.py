"""
Training Utilities

Training loop with callbacks, early stopping, and experiment tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class TrainerConfig:
    """Training configuration."""
    
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    early_stopping_patience: int = 10
    scheduler: str = "cosine"  # 'cosine', 'onecycle', 'step', 'none'
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 10
    save_best_model: bool = True
    device: str = "auto"


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Trainer for time series forecasting models.
    
    Features:
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Experiment tracking (MLflow optional)
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig = None,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None
    ):
        self.config = config or TrainerConfig()
        self.model = model
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": []}
    
    def _setup_scheduler(self, steps_per_epoch: int):
        """Set up learning rate scheduler."""
        total_steps = steps_per_epoch * self.config.max_epochs
        
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps
            )
        elif self.config.scheduler == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                total_steps=total_steps
            )
        elif self.config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            output = self.model(x)
            if isinstance(output, dict):
                pred = output.get("predictions", output.get("forecast"))
                # Handle quantile predictions
                if pred.dim() == 3:
                    pred = pred[..., 1]  # Take median
            else:
                pred = output
            
            loss = self.loss_fn(pred, y)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
            
            self.optimizer.step()
            
            if self.scheduler and self.config.scheduler in ["cosine", "onecycle"]:
                self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                if isinstance(output, dict):
                    pred = output.get("predictions", output.get("forecast"))
                    if pred.dim() == 3:
                        pred = pred[..., 1]
                else:
                    pred = output
                
                loss = self.loss_fn(pred, y)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        callbacks: List[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callbacks: Optional list of callback functions
            
        Returns:
            Training history
        """
        self._setup_scheduler(len(train_loader))
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.history["val_loss"].append(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss and self.config.save_best_model:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt")
                
                # Early stopping
                if early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Step scheduler
            if self.scheduler and self.config.scheduler == "step":
                self.scheduler.step()
            
            # Log
            if (epoch + 1) % self.config.log_every_n_steps == 0:
                msg = f"Epoch {epoch + 1}/{self.config.max_epochs} | Train Loss: {train_loss:.6f}"
                if val_loss:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_loss, val_loss)
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f}s")
        
        return self.history
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    
    def predict(
        self,
        data_loader: DataLoader
    ) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                output = self.model(x)
                if isinstance(output, dict):
                    pred = output.get("predictions", output.get("forecast"))
                    if pred.dim() == 3:
                        pred = pred[..., 1]
                else:
                    pred = output
                
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)
                
                output = self.model(x)
                if isinstance(output, dict):
                    pred = output.get("predictions", output.get("forecast"))
                    if pred.dim() == 3:
                        pred = pred[..., 1]
                else:
                    pred = output
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return {
            "MAE": np.mean(np.abs(preds - targets)),
            "RMSE": np.sqrt(np.mean((preds - targets) ** 2)),
            "MAPE": np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100
        }
