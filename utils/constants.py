"""
Constants and configurations for AQI Prediction Project
"""

# AQI Category Breakpoints (Indian Standards)
AQI_BREAKPOINTS = {
    'Good': (0, 50),
    'Satisfactory': (51, 100),
    'Moderate': (101, 200),
    'Poor': (201, 300),
    'Very Poor': (301, 400),
    'Severe': (401, 500)
}

# AQI Category Colors
AQI_COLORS = {
    'Good': '#00E400',
    'Satisfactory': '#FFFF00',
    'Moderate': '#FF7E00',
    'Poor': '#FF0000',
    'Very Poor': '#8B0000',
    'Severe': '#800080'
}

# Pollutant Columns
POLLUTANT_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# Feature columns for model training
FEATURE_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Month', 'Season']

# Target column
TARGET_COLUMN = 'AQI'

# Season mapping
SEASON_MAP = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}

# Model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# File paths (relative)
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'
MODEL_PATH = 'models/'