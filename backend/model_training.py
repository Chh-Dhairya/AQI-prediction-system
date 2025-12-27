"""
Model training module with multiple ML algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Dict, Tuple, Optional
from utils.constants import RANDOM_STATE, TEST_SIZE, CV_FOLDS

class ModelTrainer:
    """Train and compare multiple ML models for AQI prediction"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize model trainer
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> None:
        """
        Split data into training and testing sets
        
        Args:
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"✓ Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
    
    def initialize_models(self) -> None:
        """Initialize ML models for training"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        }
        print(f"✓ Initialized {len(self.models)} models")
    
    def train_models(self) -> Dict:
        """
        Train all models and evaluate performance
        
        Returns:
            Dictionary with model performance metrics
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        if not self.models:
            self.initialize_models()
        
        print("\n" + "="*60)
        print("Training Models...")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=CV_FOLDS, scoring='r2', n_jobs=-1
            )
            
            self.results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"  Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
            print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\n" + "="*60)
        return self.results
    
    def select_best_model(self) -> Tuple[str, object]:
        """
        Select best model based on test R² score
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No models trained. Call train_models() first.")
        
        best_name = max(self.results, key=lambda k: self.results[k]['test_r2'])
        self.best_model = self.results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\n✓ Best Model: {best_name}")
        print(f"  Test R²: {self.results[best_name]['test_r2']:.4f}")
        print(f"  Test RMSE: {self.results[best_name]['test_rmse']:.2f}")
        
        return best_name, self.best_model
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the best model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model() first.")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return importance
        else:
            print("Selected model does not support feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the best model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model() first.")
        
        joblib.dump(self.best_model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results
        
        Returns:
            DataFrame with model comparison
        """
        summary = []
        for name, metrics in self.results.items():
            summary.append({
                'Model': name,
                'Train RMSE': metrics['train_rmse'],
                'Test RMSE': metrics['test_rmse'],
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Test MAE': metrics['test_mae'],
                'CV R² Mean': metrics['cv_r2_mean']
            })
        
        return pd.DataFrame(summary).sort_values('Test R²', ascending=False)