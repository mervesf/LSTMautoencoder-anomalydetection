<p align="center">
  <img src="assets/banner.png" alt="LSTM Anomaly Detection" width="800"/>
</p>

<h1 align="center">🔍 LSTM Autoencoder Anomaly Detection</h1>

<p align="center">
  <strong>Deep Learning-based Time Series Anomaly Detection using PyTorch</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#results">Results</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-LSTM-red.svg" alt="LSTM"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"/>
</p>

---

## 📊 Overview

Detect anomalies in time series data using **LSTM Autoencoders**. The model learns to reconstruct normal patterns, and flags data points with high reconstruction error as anomalies.

<p align="center">
  <img src="assets/dashboard.png" alt="Detection Dashboard" width="800"/>
</p>

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **LSTM Autoencoder** | Deep learning model for sequence reconstruction |
| 📈 **Multiple Architectures** | Basic and Stacked LSTM options |
| 🎯 **Smart Thresholding** | 5 different threshold methods (Mean+Std, Percentile, IQR, MAD) |
| 📊 **Visualizations** | Training history, anomaly scores, detection dashboard |
| ⚡ **GPU Support** | CUDA acceleration for fast training |
| 🔄 **Early Stopping** | Prevent overfitting with patience-based stopping |
| 💾 **Checkpointing** | Save and load trained models |
| 🧪 **Synthetic Data** | Built-in data generator for testing |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LSTM AUTOENCODER                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input ──► [LSTM Encoder] ──► Latent Space ──► [LSTM Decoder] ──► Output
│   (seq)      (compress)        (features)       (reconstruct)     (seq)
│                                                             │
│   Anomaly Score = MSE(Input, Output)                       │
│   If score > threshold → ANOMALY                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Model Variants

| Model | Layers | Use Case |
|-------|--------|----------|
| `LSTMAutoencoder` | 1 LSTM each | Simple patterns, fast training |
| `StackedLSTMAutoencoder` | Multiple LSTM | Complex patterns, better accuracy |

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/LSTMautoencoder-anomalydetection.git
cd LSTMautoencoder-anomalydetection

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, Scikit-learn

## 📦 Project Structure

```
LSTMautoencoder-anomalydetection/
├── 📂 src/
│   ├── model.py          # LSTM Autoencoder architectures
│   ├── trainer.py        # Training pipeline
│   ├── detector.py       # Anomaly detection logic
│   ├── visualizer.py     # Visualization tools
│   └── data_generator.py # Synthetic data generation
├── 📂 notebooks/
│   └── demo.ipynb        # Interactive demonstration
├── 📂 data/
│   └── taxi_rides.csv    # Sample dataset
├── 📂 models/
│   └── (saved models)
├── 📂 output/
│   └── (generated plots)
├── requirements.txt
├── LICENSE
└── README.md
```

## 🎯 Quick Start

### Basic Usage

```python
from src import (
    LSTMAutoencoder, Trainer, AnomalyDetector,
    DataPreprocessor, AnomalyVisualizer, ThresholdMethod
)
import numpy as np

# 1. Prepare Data
preprocessor = DataPreprocessor(scaler_type='minmax')
data_scaled = preprocessor.fit_transform(your_data)

# Create sequences for LSTM
sequences = DataPreprocessor.create_sequences(data_scaled, seq_length=50)

# Split data
train_size = int(0.8 * len(sequences))
train_data = sequences[:train_size]
test_data = sequences[train_size:]

# 2. Create & Train Model
model = LSTMAutoencoder(input_size=1, hidden_size=32)
trainer = Trainer(model, learning_rate=0.001)
history = trainer.fit(train_data, val_data=test_data, epochs=50)

# 3. Detect Anomalies
detector = AnomalyDetector(model)
detector.fit(train_data, method=ThresholdMethod.MEAN_STD, n_std=2)
results = detector.detect(test_data)

print(f"Found {results.is_anomaly.sum()} anomalies ({results.anomaly_ratio:.1%})")

# 4. Visualize Results
viz = AnomalyVisualizer()
viz.create_dashboard(test_data, results, history, save_path='output/dashboard.png')
```

### Using Synthetic Data

```python
from src import TimeSeriesGenerator, generate_sample_dataset

# Generate taxi-like data with anomalies
generator = TimeSeriesGenerator(seed=42)
data, injected_anomalies = generator.generate_taxi_data(
    n_points=5000,
    anomaly_ratio=0.02
)

# Or use convenience function
data, anomalies = generate_sample_dataset('data/taxi_rides.csv')
```

## 📖 Documentation

### Threshold Methods

| Method | Formula | Best For |
|--------|---------|----------|
| `MEAN_STD` | μ + k×σ | Gaussian-like errors |
| `PERCENTILE` | Top k% | Known anomaly rate |
| `IQR` | Q3 + k×IQR | Robust to outliers |
| `MAD` | Median + k×MAD | Heavy-tailed distributions |
| `STATIC` | Fixed value | Domain knowledge |

```python
# Example: Different threshold methods
detector.fit(data, method=ThresholdMethod.MEAN_STD, n_std=2)
detector.fit(data, method=ThresholdMethod.PERCENTILE, percentile=95)
detector.fit(data, method=ThresholdMethod.IQR, k=1.5)
```

### Model Configuration

```python
# Basic model
model = LSTMAutoencoder(
    input_size=1,      # Number of features
    hidden_size=32,    # Latent dimension
    num_layers=1,      # LSTM layers
    dropout=0.0        # Regularization
)

# Stacked model (for complex patterns)
model = StackedLSTMAutoencoder(
    input_size=1,
    hidden_sizes=[64, 32],  # Progressive compression
    dropout=0.2
)
```

### Training Options

```python
trainer = Trainer(model, learning_rate=0.001, weight_decay=1e-5)
history = trainer.fit(
    train_data,
    val_data=val_data,
    epochs=100,
    batch_size=32,
    early_stopping_patience=15  # Stop if no improvement
)

# Save/Load checkpoints
trainer.save_checkpoint('models/best_model.pt')
trainer.load_checkpoint('models/best_model.pt')
```

## 📊 Results

### Sample Detection Output

```
============================================================
📋 DETECTION SUMMARY
============================================================

Total Samples:     1000
Anomalies Found:   23
Anomaly Ratio:     2.30%

Threshold:         0.012458
Mean Error:        0.004521
Max Error:         0.089234

Training Epochs:   47 (early stopped)
Final Loss:        0.003892
============================================================
```

### Anomaly Types Detected

| Type | Description | Detection Rate |
|------|-------------|----------------|
| Point | Single spikes | 95%+ |
| Collective | Sequence of unusual values | 90%+ |
| Level Shift | Sudden baseline change | 85%+ |
| Trend | Unexpected trend | 80%+ |

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber
- [Anomaly Detection with LSTM Autoencoders](https://arxiv.org/abs/1607.00148)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📬 Contact

**Merve** - [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)

Project Link: [https://github.com/yourusername/LSTMautoencoder-anomalydetection](https://github.com/yourusername/LSTMautoencoder-anomalydetection)

---

<p align="center">
  Made with ❤️ by Merve
</p>

<p align="center">
  ⭐ Star this repo if you found it useful!
</p>
