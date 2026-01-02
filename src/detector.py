"""
Anomaly Detection Module

This module provides anomaly detection capabilities using trained
LSTM Autoencoder models with various thresholding strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ThresholdMethod(Enum):
    """Available threshold calculation methods."""
    STATIC = "static"           # Fixed threshold value
    MEAN_STD = "mean_std"       # Mean + k * std
    PERCENTILE = "percentile"   # Top k percentile
    MAD = "mad"                 # Median Absolute Deviation
    IQR = "iqr"                 # Interquartile Range


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    is_anomaly: np.ndarray          # Boolean array indicating anomalies
    anomaly_scores: np.ndarray      # Reconstruction errors
    threshold: float                 # Threshold used
    anomaly_indices: np.ndarray     # Indices of anomalies
    anomaly_ratio: float            # Percentage of anomalies
    
    def __repr__(self) -> str:
        return (f"AnomalyResult(anomalies={self.is_anomaly.sum()}, "
                f"ratio={self.anomaly_ratio:.2%}, threshold={self.threshold:.6f})")


class AnomalyDetector:
    """
    Anomaly Detector using LSTM Autoencoder.
    
    Detects anomalies in time series data by measuring reconstruction
    error from a trained autoencoder. Points with high reconstruction
    error are flagged as anomalies.
    
    Args:
        model: Trained LSTMAutoencoder model
        device: Device for inference ('cuda' or 'cpu')
        
    Example:
        >>> detector = AnomalyDetector(trained_model)
        >>> detector.fit(normal_data)  # Learn threshold from normal data
        >>> results = detector.detect(new_data)
        >>> print(f"Found {results.is_anomaly.sum()} anomalies")
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        self.threshold: Optional[float] = None
        self.baseline_errors: Optional[np.ndarray] = None
        self.scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
    
    def fit(self, data: np.ndarray, 
            method: ThresholdMethod = ThresholdMethod.MEAN_STD,
            **kwargs) -> 'AnomalyDetector':
        """
        Fit the detector by calculating threshold from normal data.
        
        Args:
            data: Normal (non-anomalous) data for threshold calculation
            method: Method to calculate threshold
            **kwargs: Method-specific parameters
                - MEAN_STD: n_std (default=2) - number of standard deviations
                - PERCENTILE: percentile (default=95)
                - MAD: k (default=3) - multiplier for MAD
                - IQR: k (default=1.5) - multiplier for IQR
                
        Returns:
            self for method chaining
        """
        # Calculate reconstruction errors on normal data
        self.baseline_errors = self._calculate_errors(data)
        
        # Calculate threshold based on method
        if method == ThresholdMethod.STATIC:
            self.threshold = kwargs.get('threshold', 0.1)
            
        elif method == ThresholdMethod.MEAN_STD:
            n_std = kwargs.get('n_std', 2)
            mean = np.mean(self.baseline_errors)
            std = np.std(self.baseline_errors)
            self.threshold = mean + n_std * std
            
        elif method == ThresholdMethod.PERCENTILE:
            percentile = kwargs.get('percentile', 95)
            self.threshold = np.percentile(self.baseline_errors, percentile)
            
        elif method == ThresholdMethod.MAD:
            k = kwargs.get('k', 3)
            median = np.median(self.baseline_errors)
            mad = np.median(np.abs(self.baseline_errors - median))
            self.threshold = median + k * mad
            
        elif method == ThresholdMethod.IQR:
            k = kwargs.get('k', 1.5)
            q1 = np.percentile(self.baseline_errors, 25)
            q3 = np.percentile(self.baseline_errors, 75)
            iqr = q3 - q1
            self.threshold = q3 + k * iqr
        
        print(f"✓ Detector fitted with {method.value} method")
        print(f"  Threshold: {self.threshold:.6f}")
        print(f"  Baseline error - Mean: {np.mean(self.baseline_errors):.6f}, "
              f"Std: {np.std(self.baseline_errors):.6f}")
        
        return self
    
    def detect(self, data: np.ndarray, 
               threshold: Optional[float] = None) -> AnomalyResult:
        """
        Detect anomalies in data.
        
        Args:
            data: Data to check for anomalies
            threshold: Optional custom threshold (overrides fitted threshold)
            
        Returns:
            AnomalyResult with detection results
        """
        if threshold is None:
            if self.threshold is None:
                raise ValueError("No threshold set. Call fit() first or provide threshold.")
            threshold = self.threshold
        
        # Calculate reconstruction errors
        errors = self._calculate_errors(data)
        
        # Detect anomalies
        is_anomaly = errors > threshold
        anomaly_indices = np.where(is_anomaly)[0]
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_scores=errors,
            threshold=threshold,
            anomaly_indices=anomaly_indices,
            anomaly_ratio=is_anomaly.mean()
        )
    
    def _calculate_errors(self, data: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for data."""
        # Ensure 3D shape
        if data.ndim == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        elif data.ndim == 1:
            data = data.reshape(1, -1, 1)
        
        # Convert to tensor
        tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed = self.model(tensor)
        
        # Calculate MSE per sample
        mse = torch.mean((tensor - reconstructed) ** 2, dim=(1, 2))
        
        return mse.cpu().numpy()
    
    def get_anomaly_windows(self, results: AnomalyResult, 
                           min_window_size: int = 1) -> List[Tuple[int, int]]:
        """
        Get contiguous windows of anomalies.
        
        Args:
            results: AnomalyResult from detect()
            min_window_size: Minimum number of consecutive anomalies
            
        Returns:
            List of (start_idx, end_idx) tuples for anomaly windows
        """
        windows = []
        in_window = False
        start_idx = 0
        
        for i, is_anom in enumerate(results.is_anomaly):
            if is_anom and not in_window:
                start_idx = i
                in_window = True
            elif not is_anom and in_window:
                if i - start_idx >= min_window_size:
                    windows.append((start_idx, i - 1))
                in_window = False
        
        # Handle window at end
        if in_window and len(results.is_anomaly) - start_idx >= min_window_size:
            windows.append((start_idx, len(results.is_anomaly) - 1))
        
        return windows
    
    def score_severity(self, results: AnomalyResult) -> np.ndarray:
        """
        Calculate severity scores for anomalies (0-1 scale).
        
        Higher scores indicate more severe anomalies.
        
        Args:
            results: AnomalyResult from detect()
            
        Returns:
            Array of severity scores (0 for normal, 0-1 for anomalies)
        """
        scores = np.zeros_like(results.anomaly_scores)
        
        if results.is_anomaly.any():
            anomaly_errors = results.anomaly_scores[results.is_anomaly]
            min_err = results.threshold
            max_err = anomaly_errors.max()
            
            if max_err > min_err:
                # Normalize anomaly scores to 0-1
                normalized = (results.anomaly_scores - min_err) / (max_err - min_err)
                scores = np.clip(normalized, 0, 1)
                scores[~results.is_anomaly] = 0
        
        return scores


class DataPreprocessor:
    """
    Preprocessor for time series data.
    
    Handles scaling, sequence creation, and data preparation
    for the LSTM Autoencoder.
    """
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """Fit the scaler on data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.scaler.fit(data)
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale the data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        scaled = self.scaler.transform(data)
        
        if len(original_shape) == 1:
            scaled = scaled.flatten()
        
        return scaled
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the scaling."""
        original_shape = data.shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        unscaled = self.scaler.inverse_transform(data)
        
        if len(original_shape) == 1:
            unscaled = unscaled.flatten()
        
        return unscaled
    
    @staticmethod
    def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences for LSTM input."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples = len(data) - seq_length + 1
        n_features = data.shape[1]
        
        sequences = np.zeros((n_samples, seq_length, n_features))
        for i in range(n_samples):
            sequences[i] = data[i:i + seq_length]
        
        return sequences
