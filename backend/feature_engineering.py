"""
Feature engineering module for AQI prediction
"""

import pandas as pd
import numpy as np
from utils.constants import SEASON_MAP, POLLUTANT_COLUMNS

class FeatureEngineer:
    """Create and transform features for ML models"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            data: Preprocessed DataFrame
        """
        self.data = data.copy()
    
    def calculate_aqi(self) -> pd.DataFrame:
        """
        Calculate AQI based on pollutant concentrations (Simplified Indian AQI)
        
        Returns:
            DataFrame with AQI column added
        """
        if 'AQI' in self.data.columns:
            print("✓ AQI column already exists")
            return self.data
        
        # Simplified AQI calculation (sub-index based on concentration)
        # Using dominant pollutant approach
        
        def calculate_sub_index(pollutant, concentration):
            """Calculate sub-index for each pollutant"""
            breakpoints = {
                'PM2.5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), 
                         (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
                'PM10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
                        (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
                'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
                       (181, 280, 201, 300), (281, 400, 301, 400), (401, 600, 401, 500)],
                'SO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
                       (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)],
                'CO': [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
                      (11, 17, 201, 300), (18, 34, 301, 400), (35, 50, 401, 500)],
                'O3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
                      (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)]
            }
            
            if pollutant not in breakpoints or pd.isna(concentration):
                return 0
            
            for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints[pollutant]:
                if bp_lo <= concentration <= bp_hi:
                    aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
                    return aqi
            
            return 500  # Maximum AQI for extreme values
        
        # Calculate sub-indices
        sub_indices = pd.DataFrame()
        for pollutant in POLLUTANT_COLUMNS:
            if pollutant in self.data.columns:
                sub_indices[f'{pollutant}_AQI'] = self.data[pollutant].apply(
                    lambda x: calculate_sub_index(pollutant, x)
                )
        
        # AQI is the maximum of all sub-indices
        self.data['AQI'] = sub_indices.max(axis=1)
        print(f"✓ AQI calculated (range: {self.data['AQI'].min():.1f} - {self.data['AQI'].max():.1f})")
        
        return self.data
    
    def create_date_features(self, date_column: str = 'Date') -> pd.DataFrame:
        """
        Create date-based features
        
        Args:
            date_column: Name of the date column
            
        Returns:
            DataFrame with date features added
        """
        if date_column not in self.data.columns:
            print(f"✗ Date column '{date_column}' not found, skipping date features")
            return self.data
        
        try:
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data['Year'] = self.data[date_column].dt.year
            self.data['Month'] = self.data[date_column].dt.month
            self.data['Day'] = self.data[date_column].dt.day
            self.data['DayOfWeek'] = self.data[date_column].dt.dayofweek
            self.data['Quarter'] = self.data[date_column].dt.quarter
            
            print("✓ Date features created")
        except Exception as e:
            print(f"✗ Error creating date features: {str(e)}")
        
        return self.data
    
    def create_season_feature(self) -> pd.DataFrame:
        """
        Create season feature based on month
        
        Returns:
            DataFrame with season column added
        """
        if 'Month' not in self.data.columns:
            print("✗ Month column not found, creating default month")
            self.data['Month'] = 6  # Default to summer
        
        self.data['Season'] = self.data['Month'].map(SEASON_MAP)
        
        # Encode seasons
        season_encoding = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
        self.data['Season'] = self.data['Season'].map(season_encoding)
        
        print("✓ Season feature created")
        return self.data
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features between pollutants
        
        Returns:
            DataFrame with interaction features
        """
        if 'PM2.5' in self.data.columns and 'PM10' in self.data.columns:
            self.data['PM_Ratio'] = self.data['PM2.5'] / (self.data['PM10'] + 1)
        
        if 'NO2' in self.data.columns and 'SO2' in self.data.columns:
            self.data['NOx_SOx_Sum'] = self.data['NO2'] + self.data['SO2']
        
        print("✓ Interaction features created")
        return self.data
    
    def get_engineered_data(self) -> pd.DataFrame:
        """
        Get the feature-engineered DataFrame
        
        Returns:
            DataFrame with all engineered features
        """
        return self.data