"""
Data preprocessing and cleaning module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class DataPreprocessor:
    """Handle data cleaning and preprocessing tasks"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor
        
        Args:
            data: Raw DataFrame to preprocess
        """
        self.data = data.copy()
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        print(f"\nMissing values before handling:\n{self.data.isnull().sum()}")
        
        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy == 'mean':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        
        print(f"Missing values after handling: {self.data.isnull().sum().sum()}")
        return self.data
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed = initial_rows - len(self.data)
        print(f"✓ Removed {removed} duplicate rows")
        return self.data
    
    def handle_outliers(self, columns: list, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers using IQR method
        
        Args:
            columns: List of columns to check for outliers
            method: 'iqr' or 'clip'
            
        Returns:
            DataFrame with outliers handled
        """
        for col in columns:
            if col not in self.data.columns:
                continue
            
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'iqr':
                initial_count = len(self.data)
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
                removed = initial_count - len(self.data)
                print(f"✓ Removed {removed} outliers from {col}")
            elif method == 'clip':
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"✓ Clipped outliers in {col}")
        
        return self.data
    
    def normalize_features(self, columns: list, fit: bool = True) -> pd.DataFrame:
        """
        Normalize specified columns using StandardScaler
        
        Args:
            columns: List of columns to normalize
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with normalized columns
        """
        if fit:
            self.data[columns] = self.scaler.fit_transform(self.data[columns])
        else:
            self.data[columns] = self.scaler.transform(self.data[columns])
        
        print(f"✓ Normalized {len(columns)} feature columns")
        return self.data
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame
        
        Returns:
            Processed DataFrame
        """
        return self.data
    
    def get_scaler(self) -> StandardScaler:
        """
        Get the fitted scaler for later use
        
        Returns:
            Fitted StandardScaler object
        """
        return self.scaler