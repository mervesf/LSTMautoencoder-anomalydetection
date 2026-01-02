"""
Synthetic Data Generator for Anomaly Detection

Generates realistic time series data with injected anomalies
for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies that can be injected."""
    POINT = "point"           # Single point spike
    CONTEXTUAL = "contextual" # Value unusual for context
    COLLECTIVE = "collective" # Sequence of unusual values
    TREND = "trend"           # Sudden trend change
    LEVEL_SHIFT = "level_shift"  # Permanent level change


@dataclass
class InjectedAnomaly:
    """Information about an injected anomaly."""
    start_idx: int
    end_idx: int
    anomaly_type: AnomalyType
    magnitude: float


class TimeSeriesGenerator:
    """
    Generate synthetic time series data with optional anomalies.
    
    Creates realistic patterns similar to taxi ride data, 
    sensor readings, or web traffic.
    
    Example:
        >>> generator = TimeSeriesGenerator(seed=42)
        >>> data, anomalies = generator.generate_taxi_data(n_points=5000)
        >>> print(f"Generated {len(data)} points with {len(anomalies)} anomaly windows")
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def generate_taxi_data(self, n_points: int = 5000,
                           anomaly_ratio: float = 0.02,
                           include_anomalies: bool = True) -> Tuple[np.ndarray, List[InjectedAnomaly]]:
        """
        Generate synthetic taxi ride count data.
        
        Simulates hourly taxi ride counts with:
        - Daily seasonality (rush hours)
        - Weekly seasonality (weekday vs weekend)
        - Random noise
        - Optional anomalies
        
        Args:
            n_points: Number of data points (hours)
            anomaly_ratio: Proportion of points to be anomalous
            include_anomalies: Whether to inject anomalies
            
        Returns:
            (data array, list of injected anomalies)
        """
        # Base signal parameters
        hours = np.arange(n_points)
        
        # Daily pattern (24-hour cycle)
        daily_pattern = self._generate_daily_pattern(hours)
        
        # Weekly pattern (168-hour cycle)
        weekly_pattern = self._generate_weekly_pattern(hours)
        
        # Combine patterns
        base_signal = 100 + 50 * daily_pattern + 20 * weekly_pattern
        
        # Add trend
        trend = 0.001 * hours
        
        # Add noise
        noise = np.random.normal(0, 5, n_points)
        
        # Combine
        data = base_signal + trend + noise
        data = np.maximum(data, 0)  # Ensure non-negative
        
        # Inject anomalies
        anomalies = []
        if include_anomalies:
            data, anomalies = self._inject_anomalies(data, anomaly_ratio)
        
        return data, anomalies
    
    def generate_sensor_data(self, n_points: int = 5000,
                             anomaly_ratio: float = 0.02,
                             include_anomalies: bool = True) -> Tuple[np.ndarray, List[InjectedAnomaly]]:
        """
        Generate synthetic sensor reading data.
        
        Simulates temperature or pressure sensor with:
        - Slow oscillation
        - Random fluctuations
        - Optional anomalies (sensor failures, spikes)
        
        Args:
            n_points: Number of data points
            anomaly_ratio: Proportion of anomalous points
            include_anomalies: Whether to inject anomalies
            
        Returns:
            (data array, list of injected anomalies)
        """
        t = np.arange(n_points)
        
        # Base oscillation (slow)
        base = 50 + 10 * np.sin(2 * np.pi * t / 500)
        
        # Medium frequency component
        medium = 5 * np.sin(2 * np.pi * t / 100)
        
        # High frequency noise
        noise = np.random.normal(0, 1, n_points)
        
        # Combine
        data = base + medium + noise
        
        # Inject anomalies
        anomalies = []
        if include_anomalies:
            data, anomalies = self._inject_anomalies(data, anomaly_ratio)
        
        return data, anomalies
    
    def _generate_daily_pattern(self, hours: np.ndarray) -> np.ndarray:
        """Generate realistic daily pattern (rush hours)."""
        hour_of_day = hours % 24
        
        # Morning rush (7-9 AM)
        morning = np.exp(-((hour_of_day - 8) ** 2) / 4)
        
        # Evening rush (5-7 PM)
        evening = np.exp(-((hour_of_day - 18) ** 2) / 4)
        
        # Late night low
        night = -0.5 * np.exp(-((hour_of_day - 3) ** 2) / 8)
        
        return morning + evening + night
    
    def _generate_weekly_pattern(self, hours: np.ndarray) -> np.ndarray:
        """Generate weekly pattern (weekday vs weekend)."""
        day_of_week = (hours // 24) % 7
        
        # Lower on weekends (days 5, 6)
        weekend_factor = np.where((day_of_week >= 5), -0.3, 0.2)
        
        return weekend_factor
    
    def _inject_anomalies(self, data: np.ndarray, 
                          ratio: float) -> Tuple[np.ndarray, List[InjectedAnomaly]]:
        """
        Inject various types of anomalies into data.
        
        Args:
            data: Original data
            ratio: Proportion of anomalous points
            
        Returns:
            (modified data, list of anomaly info)
        """
        data = data.copy()
        n_points = len(data)
        n_anomalies = int(n_points * ratio)
        
        anomalies = []
        used_indices = set()
        
        # Distribute anomaly types
        anomaly_types = [
            (AnomalyType.POINT, 0.4),
            (AnomalyType.COLLECTIVE, 0.3),
            (AnomalyType.LEVEL_SHIFT, 0.2),
            (AnomalyType.TREND, 0.1)
        ]
        
        for anom_type, proportion in anomaly_types:
            count = int(n_anomalies * proportion)
            
            for _ in range(count):
                # Find unused index
                attempts = 0
                while attempts < 100:
                    idx = np.random.randint(100, n_points - 100)
                    if idx not in used_indices:
                        break
                    attempts += 1
                
                if attempts >= 100:
                    continue
                
                # Inject anomaly
                if anom_type == AnomalyType.POINT:
                    magnitude = np.random.uniform(3, 6) * np.std(data)
                    direction = np.random.choice([-1, 1])
                    data[idx] += direction * magnitude
                    used_indices.add(idx)
                    anomalies.append(InjectedAnomaly(idx, idx, anom_type, magnitude))
                    
                elif anom_type == AnomalyType.COLLECTIVE:
                    length = np.random.randint(5, 20)
                    magnitude = np.random.uniform(2, 4) * np.std(data)
                    direction = np.random.choice([-1, 1])
                    end_idx = min(idx + length, n_points)
                    data[idx:end_idx] += direction * magnitude
                    for i in range(idx, end_idx):
                        used_indices.add(i)
                    anomalies.append(InjectedAnomaly(idx, end_idx-1, anom_type, magnitude))
                    
                elif anom_type == AnomalyType.LEVEL_SHIFT:
                    magnitude = np.random.uniform(2, 4) * np.std(data)
                    direction = np.random.choice([-1, 1])
                    length = np.random.randint(20, 50)
                    end_idx = min(idx + length, n_points)
                    data[idx:end_idx] += direction * magnitude
                    for i in range(idx, end_idx):
                        used_indices.add(i)
                    anomalies.append(InjectedAnomaly(idx, end_idx-1, anom_type, magnitude))
                    
                elif anom_type == AnomalyType.TREND:
                    length = np.random.randint(10, 30)
                    end_idx = min(idx + length, n_points)
                    slope = np.random.uniform(0.5, 2) * np.random.choice([-1, 1])
                    trend = slope * np.arange(end_idx - idx)
                    data[idx:end_idx] += trend
                    magnitude = abs(trend[-1])
                    for i in range(idx, end_idx):
                        used_indices.add(i)
                    anomalies.append(InjectedAnomaly(idx, end_idx-1, anom_type, magnitude))
        
        return data, anomalies
    
    def get_anomaly_labels(self, n_points: int, 
                           anomalies: List[InjectedAnomaly]) -> np.ndarray:
        """
        Get binary labels for anomalies.
        
        Args:
            n_points: Total number of points
            anomalies: List of injected anomalies
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        labels = np.zeros(n_points, dtype=int)
        for anom in anomalies:
            labels[anom.start_idx:anom.end_idx + 1] = 1
        return labels
    
    def save_to_csv(self, data: np.ndarray, filepath: str,
                    anomalies: Optional[List[InjectedAnomaly]] = None) -> None:
        """
        Save generated data to CSV file.
        
        Args:
            data: Time series data
            filepath: Output file path
            anomalies: Optional list of anomalies for labeling
        """
        df = pd.DataFrame({'value': data})
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(data), freq='H')
        
        if anomalies:
            labels = self.get_anomaly_labels(len(data), anomalies)
            df['is_anomaly'] = labels
        
        df.to_csv(filepath, index=False)
        print(f"✓ Data saved to {filepath}")


def generate_sample_dataset(output_path: str = 'data/taxi_rides.csv',
                           n_points: int = 5000,
                           seed: int = 42) -> Tuple[np.ndarray, List[InjectedAnomaly]]:
    """
    Convenience function to generate and save sample dataset.
    
    Args:
        output_path: Path to save CSV
        n_points: Number of data points
        seed: Random seed
        
    Returns:
        (data array, anomaly list)
    """
    generator = TimeSeriesGenerator(seed=seed)
    data, anomalies = generator.generate_taxi_data(n_points=n_points)
    generator.save_to_csv(data, output_path, anomalies)
    
    print(f"Generated {n_points} points with {len(anomalies)} anomaly windows")
    return data, anomalies
