# 🌍 Air Quality Index (AQI) Prediction System

A comprehensive, production-ready data science application for predicting and analyzing Air Quality Index (AQI) in Indian cities using machine learning.

## 📋 Project Overview

This project provides an end-to-end solution for:
- Analyzing historical air quality data from Indian cities
- Predicting AQI levels using multiple machine learning algorithms
- Identifying major pollution contributors
- Providing health recommendations based on AQI categories
- Interactive visualization through a professional Streamlit web interface

## 🎯 Key Features

- **Multi-Model Training**: Implements and compares Linear Regression, Random Forest, Gradient Boosting, and XGBoost
- **Real-time Prediction**: User-friendly interface for instant AQI predictions
- **Comprehensive Analysis**: Detailed EDA with interactive visualizations
- **Health Insights**: Actionable recommendations based on AQI categories
- **Production-Ready**: Modular architecture with proper error handling and logging

## 🗂️ Project Structure

```
AQI_Prediction_Project/
│
├── data/
│   ├── raw/                      # Original datasets
│   │   └── air_quality_data.csv
│   └── processed/                # Cleaned and engineered data
│       └── processed_data.csv
│
├── backend/
│   ├── data_loader.py           # Data loading and validation
│   ├── preprocessing.py         # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature creation and AQI calculation
│   ├── model_training.py        # ML model training and comparison
│   ├── model_evaluation.py      # Model evaluation and metrics
│   └── aqi_prediction.py        # Prediction and inference logic
│
├── frontend/
│   └── app.py                   # Streamlit web application
│
├── models/
│   └── best_model.pkl           # Trained model (generated after training)
│
├── utils/
│   └── constants.py             # Project constants and configurations
│
├── requirements.txt             # Python dependencies
├── run.py                       # Main pipeline execution script
└── README.md                    # Project documentation
```

## 📊 Dataset Description

### Input Features
- **PM2.5**: Particulate Matter ≤ 2.5 μm (µg/m³)
- **PM10**: Particulate Matter ≤ 10 μm (µg/m³)
- **NO₂**: Nitrogen Dioxide (µg/m³)
- **SO₂**: Sulfur Dioxide (µg/m³)
- **CO**: Carbon Monoxide (mg/m³)
- **O₃**: Ozone (µg/m³)
- **Temporal Features**: Month, Season, Day of Week
- **Location**: City information

### Target Variable
- **AQI**: Air Quality Index (0-500 scale following Indian standards)

### AQI Categories (Indian Standards)
| Category | Range | Color | Health Impact |
|----------|-------|-------|---------------|
| Good | 0-50 | Green | Minimal impact |
| Satisfactory | 51-100 | Yellow | Minor discomfort to sensitive people |
| Moderate | 101-200 | Orange | Breathing discomfort to people with respiratory issues |
| Poor | 201-300 | Red | Breathing discomfort to most people |
| Very Poor | 301-400 | Purple | Respiratory illness on prolonged exposure |
| Severe | 401-500 | Maroon | Affects healthy people, serious impact on existing conditions |

## 🛠️ Tech Stack

- **Programming Language**: Python 3.8+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Joblib

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd AQI_Prediction_Project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the ML Pipeline
```bash
python run.py
```

This will:
- Create necessary directories
- Generate sample data (if not present)
- Preprocess and clean the data
- Engineer features
- Train multiple ML models
- Evaluate and select the best model
- Save the trained model

Expected output:
```
AIR QUALITY INDEX (AQI) PREDICTION - ML PIPELINE
================================================================

Step 1: Setting up project structure...
✓ Directory structure created

Step 2: Loading data...
✓ Data loaded successfully: 5000 rows, 9 columns
...

✓ PIPELINE COMPLETED SUCCESSFULLY
Trained model saved to: models/best_model.pkl
```

### Step 5: Launch the Web Application
```bash
streamlit run frontend/app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📱 Using the Application

### 1. Data Analysis Page
- View dataset statistics and sample data
- Explore pollutant distributions
- Analyze city-wise AQI variations
- Examine correlation between pollutants

### 2. AQI Prediction Page
- Input pollutant concentrations using sliders
- Select month and season
- Get instant AQI prediction
- View health category and recommendations
- See visual gauge representation

### 3. Pollution Insights Page
- Seasonal AQI trends
- Major pollution contributors
- AQI category distribution
- City-wise risk analysis

### 4. About Page
- Project information
- Technical details
- AQI category definitions

## 🤖 Machine Learning Pipeline

### 1. Data Loading & Validation
```python
from backend.data_loader import DataLoader

loader = DataLoader('data/raw/air_quality_data.csv')
data = loader.load_data()
loader.validate_data()
```

### 2. Preprocessing
- Handle missing values (median imputation)
- Remove duplicates
- Outlier handling (IQR method)
- Feature scaling (StandardScaler)

### 3. Feature Engineering
- AQI calculation from pollutant concentrations
- Temporal features (Month, Season, Quarter)
- Interaction features (PM ratios, pollutant sums)
- Categorical encoding

### 4. Model Training
Trains and compares four models:
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble learning with 100 trees
- **Gradient Boosting**: Sequential ensemble with boosting
- **XGBoost**: Optimized gradient boosting

### 5. Model Evaluation
Metrics used:
- **RMSE** (Root Mean Squared Error): Prediction error magnitude
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Model fit quality
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **Cross-Validation**: 5-fold CV for robustness

### 6. Model Selection
Best model is selected based on highest test R² score, considering:
- Predictive accuracy
- Generalization capability
- Cross-validation performance

## 📈 Model Performance

Typical performance metrics (on sample data):

| Model | Test RMSE | Test R² | Test MAE | CV R² Mean |
|-------|-----------|---------|----------|------------|
| XGBoost | 15-20 | 0.95-0.98 | 10-15 | 0.94-0.97 |
| Random Forest | 18-23 | 0.93-0.96 | 12-17 | 0.92-0.95 |
| Gradient Boosting | 20-25 | 0.91-0.95 | 14-19 | 0.90-0.94 |
| Linear Regression | 25-35 | 0.85-0.90 | 18-25 | 0.84-0.89 |

*Note: Actual performance depends on your dataset quality and size*

## 🎨 Screenshots

### Data Analysis Dashboard
*(Screenshot placeholder - Add after running the application)*

### AQI Prediction Interface
*(Screenshot placeholder - Add after running the application)*

### Pollution Insights
*(Screenshot placeholder - Add after running the application)*

## 🔧 Customization

### Using Your Own Dataset

1. Place your CSV file in `data/raw/`
2. Ensure it contains required columns: PM2.5, PM10, NO2, SO2, CO, O3
3. Update the file path in `run.py`:
```python
data_path = 'data/raw/your_dataset.csv'
```

### Adjusting Model Parameters

Edit `backend/model_training.py` to modify hyperparameters:
```python
'Random Forest': RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples for split
    random_state=RANDOM_STATE
)
```

### Modifying AQI Breakpoints

Update `utils/constants.py` to change AQI thresholds:
```python
AQI_BREAKPOINTS = {
    'Good': (0, 50),
    'Satisfactory': (51, 100),
    # Add more categories
}
```

## 📝 Code Quality Features

- **Modular Architecture**: Separated concerns with clear module boundaries
- **Type Hints**: Enhanced code readability and IDE support
- **Docstrings**: Comprehensive documentation for all functions
- **Error Handling**: Try-except blocks with informative messages
- **Constants Management**: Centralized configuration
- **Logging**: Progress indicators and status messages
- **PEP 8 Compliance**: Clean, readable Python code

## 🐛 Troubleshooting

### Issue: Model not found error
**Solution**: Run `python run.py` first to train and save the model

### Issue: Import errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Data file not found
**Solution**: The script generates sample data automatically if no data exists

### Issue: Streamlit not opening
**Solution**: Check if port 8501 is available or specify a different port:
```bash
streamlit run frontend/app.py --server.port 8502
```

## 🤝 Contributing

This project is designed as an internship-grade demonstration. For improvements:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 📄 License

This project is available for educational and demonstration purposes.

## 👨‍💻 Developer

Built with focus on:
- Clean code architecture
- ML best practices
- Professional UI/UX design
- Production-ready implementation
- Comprehensive documentation

## 📧 Contact

For questions, feedback, or collaboration opportunities, please reach out through the project repository.

## 🙏 Acknowledgments

- Indian Central Pollution Control Board (CPCB) for AQI standards
- Scikit-learn and XGBoost teams for excellent ML libraries
- Streamlit for the intuitive web framework
- Open-source community for various tools and libraries

---

**Note**: This project is created for educational and internship demonstration purposes. For production deployment, consider additional security, scalability, and monitoring features.
