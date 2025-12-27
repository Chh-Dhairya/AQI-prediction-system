"""
Model evaluation and visualization module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple

class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
            'MAE': mean_absolute_error(self.y_test, self.y_pred),
            'R²': r2_score(self.y_test, self.y_pred),
            'MAPE': np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        }
        
        return metrics
    
    def plot_predictions(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Create prediction visualization plots
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Actual vs Predicted scatter
        axes[0].scatter(self.y_test, self.y_pred, alpha=0.5, s=20)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual AQI', fontsize=12)
        axes[0].set_ylabel('Predicted AQI', fontsize=12)
        axes[0].set_title('Actual vs Predicted AQI', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = self.y_test - self.y_pred
        axes[1].scatter(self.y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted AQI', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        """
        Plot distribution of prediction errors
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        errors = self.y_test - self.y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Prediction Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel('Prediction Error', fontsize=12)
        axes[1].set_title('Error Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 10, 
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot feature importance if available
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure object or None
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def get_performance_summary(self) -> str:
        """
        Get text summary of model performance
        
        Returns:
            Formatted string with metrics
        """
        metrics = self.calculate_metrics()
        
        summary = "Model Performance Summary\n"
        summary += "=" * 40 + "\n"
        for metric, value in metrics.items():
            if metric == 'MAPE':
                summary += f"{metric}: {value:.2f}%\n"
            else:
                summary += f"{metric}: {value:.4f}\n"
        
        return summary