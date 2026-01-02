"""
LSTM Autoencoder Anomaly Detection

A PyTorch-based toolkit for detecting anomalies in time series data
using LSTM Autoencoders.
"""

from .model import LSTMAutoencoder, StackedLSTMAutoencoder, create_model
from .trainer import Trainer, EarlyStopping, create_sequences, train_test_split_ts
from .detector import AnomalyDetector, DataPreprocessor, ThresholdMethod, AnomalyResult
from .visualizer import AnomalyVisualizer
from .data_generator import TimeSeriesGenerator, generate_sample_dataset

__version__ = '1.0.0'
__author__ = 'Merve'
__all__ = [
    'LSTMAutoencoder',
    'StackedLSTMAutoencoder', 
    'create_model',
    'Trainer',
    'EarlyStopping',
    'AnomalyDetector',
    'DataPreprocessor',
    'ThresholdMethod',
    'AnomalyResult',
    'AnomalyVisualizer',
    'TimeSeriesGenerator',
    'generate_sample_dataset',
    'create_sequences',
    'train_test_split_ts'
]
