"""
Training Pipeline for LSTM Autoencoder

This module provides training, validation, and model management utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from pathlib import Path
import json
from datetime import datetime

try:
    from .model import LSTMAutoencoder, create_model
except ImportError:
    from model import LSTMAutoencoder, create_model


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        restore_best: Whether to restore best weights on stop
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() 
                                    for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.should_stop


class Trainer:
    """
    Training manager for LSTM Autoencoder models.
    
    Handles the complete training pipeline including:
    - Training loop with progress tracking
    - Validation
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Training history
    
    Args:
        model: The autoencoder model to train
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        device: Device to train on ('cuda' or 'cpu')
        
    Example:
        >>> model = LSTMAutoencoder(input_size=1, hidden_size=32)
        >>> trainer = Trainer(model, learning_rate=0.001)
        >>> history = trainer.fit(train_data, val_data, epochs=100)
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, device: Optional[str] = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def fit(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            early_stopping_patience: int = 15,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Training data of shape (samples, seq_len, features)
            val_data: Validation data (optional)
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping (0 to disable)
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Prepare data loaders
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size) if val_data is not None else None
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
        
        if verbose:
            print(f"Training on {self.device}")
            print(f"Train samples: {len(train_data)}, Epochs: {epochs}, Batch size: {batch_size}")
            print("-" * 60)
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                if verbose:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f}")
                
                # Early stopping check
                if early_stopping and early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {train_loss:.6f}")
        
        if verbose:
            print("-" * 60)
            print(f"Training complete. Final loss: {self.history['train_loss'][-1]:.6f}")
        
        return self.history
    
    def _create_dataloader(self, data: np.ndarray, batch_size: int,
                           shuffle: bool = False) -> DataLoader:
        """Create a DataLoader from numpy array."""
        if data is None:
            return None
        
        # Ensure 3D shape (samples, seq_len, features)
        if data.ndim == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        
        tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor, tensor)  # Input = target for autoencoder
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers
            },
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {filepath}")


def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Create sequences from time series data for LSTM input.
    
    Args:
        data: 1D or 2D time series data
        seq_length: Length of each sequence
        
    Returns:
        Array of shape (num_sequences, seq_length, features)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    
    return np.array(sequences)


def train_test_split_ts(data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series data preserving temporal order.
    
    Args:
        data: Time series data
        train_ratio: Proportion of data for training
        
    Returns:
        (train_data, test_data)
    """
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]
