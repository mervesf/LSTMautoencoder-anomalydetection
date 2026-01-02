"""
LSTM Autoencoder Model for Anomaly Detection

This module contains the neural network architecture for time series
anomaly detection using LSTM-based autoencoders.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder that compresses input sequences into a latent representation.
    
    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability (applied if num_layers > 1)
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, dropout: float = 0.0):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            outputs: LSTM outputs (batch, seq_len, hidden_size)
            (hidden, cell): Final hidden and cell states
        """
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder that reconstructs sequences from latent representation.
    
    Args:
        hidden_size: Number of features in hidden state (input from encoder)
        output_size: Number of features in output (should match original input)
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super(LSTMDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Reconstructed tensor of shape (batch, seq_len, output_size)
        """
        decoded, _ = self.lstm(x)
        output = self.output_layer(decoded)
        return output


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for Time Series Anomaly Detection.
    
    The model learns to reconstruct normal time series patterns.
    High reconstruction error indicates anomalous behavior.
    
    Architecture:
        Input -> LSTM Encoder -> Latent Space -> LSTM Decoder -> Reconstruction
    
    Args:
        input_size: Number of input features
        hidden_size: Size of the latent representation
        num_layers: Number of LSTM layers in encoder/decoder
        dropout: Dropout probability for regularization
        
    Example:
        >>> model = LSTMAutoencoder(input_size=1, hidden_size=32)
        >>> x = torch.randn(16, 100, 1)  # batch=16, seq_len=100, features=1
        >>> reconstructed = model(x)
        >>> loss = nn.MSELoss()(reconstructed, x)
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.0):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            hidden_size=hidden_size,
            output_size=input_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        encoded, _ = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the encoded (latent) representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation
        """
        encoded, _ = self.encoder(x)
        return encoded
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error for input.
        
        Args:
            x: Input tensor
            
        Returns:
            MSE reconstruction error per sample
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse


class StackedLSTMAutoencoder(nn.Module):
    """
    Deeper LSTM Autoencoder with multiple encoding layers.
    
    Provides better feature extraction for complex time series.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden sizes for each layer (e.g., [64, 32])
        dropout: Dropout probability
    """
    
    def __init__(self, input_size: int = 1, 
                 hidden_sizes: list = [64, 32],
                 dropout: float = 0.2):
        super(StackedLSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build encoder layers
        encoder_layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(
                nn.LSTM(in_size, hidden_size, batch_first=True)
            )
            in_size = hidden_size
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Build decoder layers (reverse order)
        decoder_layers = []
        decoder_sizes = hidden_sizes[::-1] + [input_size]
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(
                nn.LSTM(decoder_sizes[i], decoder_sizes[i+1], batch_first=True)
            )
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked autoencoder."""
        # Encode
        encoded = x
        for lstm in self.encoder_layers:
            encoded, _ = lstm(encoded)
            encoded = self.dropout(encoded)
        
        # Decode
        decoded = encoded
        for lstm in self.decoder_layers:
            decoded, _ = lstm(decoded)
        
        return decoded


def create_model(model_type: str = 'basic', **kwargs) -> nn.Module:
    """
    Factory function to create autoencoder models.
    
    Args:
        model_type: Type of model ('basic' or 'stacked')
        **kwargs: Model parameters
        
    Returns:
        Instantiated model
    """
    if model_type == 'basic':
        return LSTMAutoencoder(**kwargs)
    elif model_type == 'stacked':
        return StackedLSTMAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
