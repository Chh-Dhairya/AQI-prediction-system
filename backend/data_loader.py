"""
Data loading and validation module
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

class DataLoader:
    """Handle data loading and initial validation"""
    
    def __init__(self, file_path: str):
        """
        Initialize DataLoader
        
        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data into pandas DataFrame
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Data file not found at {self.file_path}")
            
            self.data = pd.read_csv(self.file_path)
            print(f"✓ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate data structure and required columns
        
        Returns:
            True if validation passes, False otherwise
        """
        if self.data is None:
            print("✗ No data loaded")
            return False
        
        required_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False
        
        print("✓ Data validation passed")
        return True
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            return {}
        
        info = {
            'rows': self.data.shape[0],
            'columns': self.data.shape[1],
            'missing_values': self.data.isnull().sum().sum(),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return info
    
    def get_column_types(self) -> pd.Series:
        """
        Get data types of all columns
        
        Returns:
            Series with column names and their data types
        """
        if self.data is None:
            return pd.Series()
        
        return self.data.dtypes