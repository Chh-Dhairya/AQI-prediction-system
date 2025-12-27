"""
Main execution script for AQI Prediction Project
Run this script to train models and prepare data
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from backend.data_loader import DataLoader
from backend.preprocessing import DataPreprocessor
from backend.feature_engineering import FeatureEngineer
from backend.model_training import ModelTrainer
from backend.model_evaluation import ModelEvaluator
from utils.constants import FEATURE_COLUMNS, TARGET_COLUMN, POLLUTANT_COLUMNS

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'frontend',
        'backend',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directory structure created")

def generate_sample_data():
    """Generate sample dataset if no data exists"""
    raw_data_path = 'data/raw/air_quality_data.csv'
    
    if os.path.exists(raw_data_path):
        print(f"✓ Data file already exists at {raw_data_path}")
        return raw_data_path
    
    print("Generating sample dataset...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic air quality data
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 
              'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow']
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    data = pd.DataFrame({
        'Date': np.random.choice(dates, n_samples),
        'City': np.random.choice(cities, n_samples),
        'PM2.5': np.random.gamma(4, 15, n_samples),  # Realistic distribution
        'PM10': np.random.gamma(5, 20, n_samples),
        'NO2': np.random.gamma(3, 10, n_samples),
        'SO2': np.random.gamma(2, 8, n_samples),
        'CO': np.random.gamma(1.5, 1, n_samples),
        'O3': np.random.gamma(3, 12, n_samples)
    })
    
    # Clip to realistic ranges
    data['PM2.5'] = data['PM2.5'].clip(5, 300)
    data['PM10'] = data['PM10'].clip(10, 500)
    data['NO2'] = data['NO2'].clip(5, 200)
    data['SO2'] = data['SO2'].clip(2, 150)
    data['CO'] = data['CO'].clip(0.1, 30)
    data['O3'] = data['O3'].clip(5, 200)
    
    # Add some seasonal patterns
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    
    # Winter months (Nov-Feb) tend to have higher pollution
    winter_mask = data['Month'].isin([11, 12, 1, 2])
    data.loc[winter_mask, 'PM2.5'] *= 1.5
    data.loc[winter_mask, 'PM10'] *= 1.4
    
    # Save to CSV
    data.to_csv(raw_data_path, index=False)
    print(f"✓ Sample dataset generated: {n_samples} records")
    print(f"✓ Saved to {raw_data_path}")
    
    return raw_data_path

def run_pipeline():
    """Execute the complete ML pipeline"""
    
    print("\n" + "="*70)
    print("AIR QUALITY INDEX (AQI) PREDICTION - ML PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Create directories
    print("Step 1: Setting up project structure...")
    create_directories()
    
    # Step 2: Generate or load data
    print("\nStep 2: Loading data...")
    data_path = generate_sample_data()
    
    loader = DataLoader(data_path)
    data = loader.load_data()
    
    if not loader.validate_data():
        print("✗ Data validation failed. Exiting...")
        return
    
    print("\nDataset Info:")
    for key, value in loader.get_basic_info().items():
        print(f"  {key}: {value}")
    
    # Step 3: Preprocessing
    print("\nStep 3: Preprocessing data...")
    preprocessor = DataPreprocessor(data)
    data = preprocessor.handle_missing_values(strategy='median')
    data = preprocessor.remove_duplicates()
    data = preprocessor.handle_outliers(POLLUTANT_COLUMNS, method='clip')
    
    # Step 4: Feature Engineering
    print("\nStep 4: Engineering features...")
    engineer = FeatureEngineer(data)
    data = engineer.create_date_features(date_column='Date')
    data = engineer.create_season_feature()
    data = engineer.calculate_aqi()
    data = engineer.create_interaction_features()
    
    # Save processed data
    processed_path = 'data/processed/processed_data.csv'
    data.to_csv(processed_path, index=False)
    print(f"✓ Processed data saved to {processed_path}")
    
    # Step 5: Prepare features for training
    print("\nStep 5: Preparing features for model training...")
    
    feature_cols = [col for col in FEATURE_COLUMNS if col in data.columns]
    
    if TARGET_COLUMN not in data.columns:
        print(f"✗ Target column '{TARGET_COLUMN}' not found. Exiting...")
        return
    
    X = data[feature_cols].copy()
    y = data[TARGET_COLUMN].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    print(f"  Features: {X.shape[1]} columns, {X.shape[0]} samples")
    print(f"  Target: {y.shape[0]} samples")
    print(f"  Feature columns: {', '.join(feature_cols)}")
    
    # Step 6: Train models
    print("\nStep 6: Training machine learning models...")
    trainer = ModelTrainer(X, y)
    trainer.split_data()
    trainer.initialize_models()
    results = trainer.train_models()
    
    # Step 7: Select and save best model
    print("\nStep 7: Selecting best model...")
    best_name, best_model = trainer.select_best_model()
    
    model_path = 'models/best_model.pkl'
    trainer.save_model(model_path)
    
    # Step 8: Model evaluation
    print("\nStep 8: Evaluating model performance...")
    evaluator = ModelEvaluator(best_model, trainer.X_test, trainer.y_test)
    
    metrics = evaluator.calculate_metrics()
    print("\nFinal Model Metrics:")
    for metric, value in metrics.items():
        if metric == 'MAPE':
            print(f"  {metric}: {value:.2f}%")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\nTop 5 Most Important Features:")
        importance = trainer.get_feature_importance(top_n=5)
        for idx, row in importance.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Step 9: Display results summary
    print("\n" + "="*70)
    print("Model Comparison Summary:")
    print("="*70)
    summary = trainer.get_results_summary()
    print(summary.to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nTrained model saved to: {model_path}")
    print(f"Processed data saved to: {processed_path}")
    print("\nTo run the Streamlit app, execute:")
    print("  streamlit run frontend/app.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n✗ Error in pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)