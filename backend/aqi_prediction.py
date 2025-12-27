"""
AQI prediction and inference module
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple
from utils.constants import AQI_BREAKPOINTS, AQI_COLORS

class AQIPredictor:
    """Handle AQI predictions and categorization"""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            self.model = None
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict AQI for given features
        
        Args:
            features: DataFrame with feature values
            
        Returns:
            Array of predicted AQI values
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(features)
        return np.clip(predictions, 0, 500)  # AQI range: 0-500
    
    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """
        Predict AQI for a single input
        
        Args:
            feature_dict: Dictionary with feature names and values
            
        Returns:
            Predicted AQI value
        """
        features_df = pd.DataFrame([feature_dict])
        prediction = self.predict(features_df)
        return float(prediction[0])
    
    @staticmethod
    def categorize_aqi(aqi_value: float) -> Tuple[str, str]:
        """
        Categorize AQI value into health category
        
        Args:
            aqi_value: AQI numeric value
            
        Returns:
            Tuple of (category_name, color_code)
        """
        for category, (lower, upper) in AQI_BREAKPOINTS.items():
            if lower <= aqi_value <= upper:
                return category, AQI_COLORS[category]
        
        return 'Severe', AQI_COLORS['Severe']
    
    @staticmethod
    def get_health_message(category: str) -> str:
        """
        Get health advisory message for AQI category
        
        Args:
            category: AQI category name
            
        Returns:
            Health advisory message
        """
        messages = {
            'Good': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'Satisfactory': 'Air quality is acceptable. However, sensitive individuals should consider limiting prolonged outdoor exertion.',
            'Moderate': 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.',
            'Poor': 'Everyone may begin to experience health effects. Members of sensitive groups may experience more serious health effects.',
            'Very Poor': 'Health alert: The risk of health effects is increased for everyone.',
            'Severe': 'Health warning of emergency conditions. The entire population is more likely to be affected.'
        }
        
        return messages.get(category, 'Unknown air quality level.')
    
    @staticmethod
    def get_recommendations(category: str) -> list:
        """
        Get recommendations based on AQI category
        
        Args:
            category: AQI category name
            
        Returns:
            List of recommendations
        """
        recommendations = {
            'Good': [
                'Enjoy outdoor activities',
                'No precautions needed'
            ],
            'Satisfactory': [
                'Unusually sensitive people should limit prolonged outdoor activities',
                'Keep windows open for ventilation'
            ],
            'Moderate': [
                'Sensitive groups should reduce prolonged outdoor exertion',
                'Consider wearing a mask during outdoor activities',
                'Monitor air quality regularly'
            ],
            'Poor': [
                'Everyone should reduce prolonged outdoor exertion',
                'Wear N95 masks when going outside',
                'Keep indoor air clean',
                'Avoid outdoor exercise'
            ],
            'Very Poor': [
                'Everyone should avoid all outdoor activities',
                'Stay indoors with windows closed',
                'Use air purifiers if available',
                'Wear N95/N99 masks if you must go outside'
            ],
            'Severe': [
                'Remain indoors and keep activity levels low',
                'Run air purifiers on high',
                'Seal windows and doors',
                'Seek medical attention if you experience symptoms'
            ]
        }
        
        return recommendations.get(category, ['Monitor air quality updates'])
    
    def predict_with_details(self, feature_dict: Dict[str, float]) -> Dict:
        """
        Predict AQI and return comprehensive details
        
        Args:
            feature_dict: Dictionary with feature values
            
        Returns:
            Dictionary with prediction details
        """
        aqi = self.predict_single(feature_dict)
        category, color = self.categorize_aqi(aqi)
        message = self.get_health_message(category)
        recommendations = self.get_recommendations(category)
        
        return {
            'aqi': round(aqi, 2),
            'category': category,
            'color': color,
            'health_message': message,
            'recommendations': recommendations
        }