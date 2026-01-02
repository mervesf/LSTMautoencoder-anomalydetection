"""
Visualization Module for Anomaly Detection

Provides comprehensive visualizations for time series anomaly detection
including training history, reconstruction errors, and anomaly highlighting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

try:
    from .detector import AnomalyResult
except ImportError:
    from detector import AnomalyResult


class AnomalyVisualizer:
    """
    Visualization toolkit for LSTM Autoencoder anomaly detection.
    
    Provides publication-ready visualizations for:
    - Training history
    - Reconstruction errors
    - Anomaly detection results
    - Time series with highlighted anomalies
    
    Example:
        >>> viz = AnomalyVisualizer()
        >>> viz.plot_training_history(history)
        >>> viz.plot_anomalies(data, results)
    """
    
    # Color scheme
    COLORS = {
        'primary': '#3498db',
        'secondary': '#2ecc71',
        'anomaly': '#e74c3c',
        'warning': '#f39c12',
        'background': '#ecf0f1',
        'grid': '#bdc3c7'
    }
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: Tuple[int, int] = (14, 5)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.figsize = figsize
        self._setup_style(style)
    
    def _setup_style(self, style: str) -> None:
        """Configure matplotlib style."""
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-v0_8')
        
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation loss over epochs.
        
        Args:
            history: Dictionary with 'train_loss' and optionally 'val_loss'
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 
                color=self.COLORS['primary'], linewidth=2, 
                label='Training Loss', marker='o', markersize=3)
        
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], 
                    color=self.COLORS['secondary'], linewidth=2,
                    label='Validation Loss', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training History')
        ax.legend()
        ax.set_xlim(1, len(epochs))
        
        # Add min loss annotation
        min_loss = min(history['train_loss'])
        min_epoch = history['train_loss'].index(min_loss) + 1
        ax.annotate(f'Min: {min_loss:.6f}', 
                   xy=(min_epoch, min_loss),
                   xytext=(min_epoch + 5, min_loss + 0.001),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        return fig
    
    def plot_reconstruction(self, original: np.ndarray, reconstructed: np.ndarray,
                           title: str = "Original vs Reconstructed",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot original and reconstructed time series.
        
        Args:
            original: Original time series
            reconstructed: Reconstructed time series from autoencoder
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 8), sharex=True)
        
        # Flatten if needed
        original = original.flatten() if original.ndim > 1 else original
        reconstructed = reconstructed.flatten() if reconstructed.ndim > 1 else reconstructed
        
        x = np.arange(len(original))
        
        # Plot original
        axes[0].plot(x, original, color=self.COLORS['primary'], linewidth=1.5, label='Original')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Original Time Series')
        axes[0].legend()
        
        # Plot reconstructed
        axes[1].plot(x, reconstructed, color=self.COLORS['secondary'], linewidth=1.5, label='Reconstructed')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Reconstructed Time Series')
        axes[1].legend()
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        return fig
    
    def plot_anomaly_scores(self, results: AnomalyResult,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot anomaly scores with threshold.
        
        Args:
            results: AnomalyResult from detector
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(results.anomaly_scores))
        
        # Plot scores
        colors = [self.COLORS['anomaly'] if a else self.COLORS['primary'] 
                  for a in results.is_anomaly]
        ax.scatter(x, results.anomaly_scores, c=colors, alpha=0.6, s=20)
        
        # Plot threshold line
        ax.axhline(y=results.threshold, color=self.COLORS['warning'], 
                   linestyle='--', linewidth=2, label=f'Threshold: {results.threshold:.4f}')
        
        # Fill anomaly region
        ax.fill_between(x, results.threshold, results.anomaly_scores.max() * 1.1,
                        alpha=0.1, color=self.COLORS['anomaly'])
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Reconstruction Error (MSE)')
        ax.set_title(f'Anomaly Scores (Detected: {results.is_anomaly.sum()} anomalies, '
                    f'{results.anomaly_ratio:.1%})')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        return fig
    
    def plot_time_series_with_anomalies(self, data: np.ndarray, results: AnomalyResult,
                                        timestamps: Optional[np.ndarray] = None,
                                        title: str = "Time Series with Anomalies",
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series with anomalies highlighted.
        
        Args:
            data: Time series data
            results: AnomalyResult from detector
            timestamps: Optional timestamp array for x-axis
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 10), 
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        data = data.flatten() if data.ndim > 1 else data
        x = timestamps if timestamps is not None else np.arange(len(data))
        
        # Top plot: Time series with anomalies
        ax1 = axes[0]
        ax1.plot(x, data, color=self.COLORS['primary'], linewidth=1, alpha=0.8, label='Normal')
        
        # Highlight anomalies
        if results.is_anomaly.any():
            anomaly_x = x[results.is_anomaly]
            anomaly_y = data[results.is_anomaly]
            ax1.scatter(anomaly_x, anomaly_y, color=self.COLORS['anomaly'], 
                       s=50, zorder=5, label='Anomaly', edgecolors='white', linewidths=0.5)
        
        ax1.set_ylabel('Value')
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        
        # Add anomaly statistics
        stats_text = f"Anomalies: {results.is_anomaly.sum()} ({results.anomaly_ratio:.1%})"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Bottom plot: Reconstruction error
        ax2 = axes[1]
        ax2.fill_between(x, 0, results.anomaly_scores, alpha=0.3, color=self.COLORS['primary'])
        ax2.plot(x, results.anomaly_scores, color=self.COLORS['primary'], linewidth=1)
        ax2.axhline(y=results.threshold, color=self.COLORS['anomaly'], 
                   linestyle='--', linewidth=2, label=f'Threshold')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Recon. Error')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        return fig
    
    def plot_error_distribution(self, results: AnomalyResult,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of reconstruction errors.
        
        Args:
            results: AnomalyResult from detector
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(results.anomaly_scores, bins=50, color=self.COLORS['primary'], 
                 alpha=0.7, edgecolor='white')
        ax1.axvline(x=results.threshold, color=self.COLORS['anomaly'], 
                   linestyle='--', linewidth=2, label=f'Threshold: {results.threshold:.4f}')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()
        
        # Box plot
        ax2 = axes[1]
        normal_errors = results.anomaly_scores[~results.is_anomaly]
        anomaly_errors = results.anomaly_scores[results.is_anomaly]
        
        data_to_plot = [normal_errors]
        labels = ['Normal']
        if len(anomaly_errors) > 0:
            data_to_plot.append(anomaly_errors)
            labels.append('Anomaly')
        
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = [self.COLORS['primary'], self.COLORS['anomaly']]
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('Error Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        return fig
    
    def create_dashboard(self, data: np.ndarray, results: AnomalyResult,
                        history: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive anomaly detection dashboard.
        
        Args:
            data: Original time series
            results: AnomalyResult from detector
            history: Optional training history
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        if history:
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        else:
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        data = data.flatten() if data.ndim > 1 else data
        x = np.arange(len(data))
        
        # 1. Time series with anomalies (top, full width)
        if history:
            ax1 = fig.add_subplot(gs[0, :])
        else:
            ax1 = fig.add_subplot(gs[0, :])
        
        ax1.plot(x, data, color=self.COLORS['primary'], linewidth=1, alpha=0.8)
        if results.is_anomaly.any():
            ax1.scatter(x[results.is_anomaly], data[results.is_anomaly], 
                       color=self.COLORS['anomaly'], s=30, zorder=5, label='Anomaly')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title('📈 Time Series with Detected Anomalies')
        ax1.legend()
        
        # 2. Anomaly scores
        if history:
            ax2 = fig.add_subplot(gs[1, 0])
        else:
            ax2 = fig.add_subplot(gs[1, 0])
        
        colors = [self.COLORS['anomaly'] if a else self.COLORS['primary'] 
                  for a in results.is_anomaly]
        ax2.scatter(x, results.anomaly_scores, c=colors, alpha=0.5, s=15)
        ax2.axhline(y=results.threshold, color=self.COLORS['warning'], linestyle='--', linewidth=2)
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('🎯 Anomaly Scores')
        
        # 3. Error distribution
        if history:
            ax3 = fig.add_subplot(gs[1, 1])
        else:
            ax3 = fig.add_subplot(gs[1, 1])
        
        ax3.hist(results.anomaly_scores, bins=40, color=self.COLORS['primary'], 
                 alpha=0.7, edgecolor='white')
        ax3.axvline(x=results.threshold, color=self.COLORS['anomaly'], 
                   linestyle='--', linewidth=2)
        ax3.set_xlabel('Reconstruction Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('📊 Error Distribution')
        
        # 4. Training history (if provided)
        if history:
            ax4 = fig.add_subplot(gs[2, 0])
            epochs = range(1, len(history['train_loss']) + 1)
            ax4.plot(epochs, history['train_loss'], color=self.COLORS['primary'], 
                    linewidth=2, label='Train')
            if 'val_loss' in history and history['val_loss']:
                ax4.plot(epochs, history['val_loss'], color=self.COLORS['secondary'], 
                        linewidth=2, label='Val')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('📉 Training History')
            ax4.legend()
            
            # 5. Summary stats
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.axis('off')
            
            stats_text = f"""
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📋 DETECTION SUMMARY
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            Total Samples:     {len(data):,}
            Anomalies Found:   {results.is_anomaly.sum():,}
            Anomaly Ratio:     {results.anomaly_ratio:.2%}
            
            Threshold:         {results.threshold:.6f}
            Mean Error:        {np.mean(results.anomaly_scores):.6f}
            Max Error:         {np.max(results.anomaly_scores):.6f}
            
            Training Epochs:   {len(history['train_loss'])}
            Final Loss:        {history['train_loss'][-1]:.6f}
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('🔍 LSTM Autoencoder Anomaly Detection Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Dashboard saved to {save_path}")
        
        return fig
