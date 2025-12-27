"""
Streamlit Web Application for AQI Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.aqi_prediction import AQIPredictor
from utils.constants import POLLUTANT_COLUMNS, AQI_COLORS, AQI_BREAKPOINTS

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .aqi-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data function
@st.cache_data
def load_sample_data():
    """Create or load sample data for demonstration"""
    try:
        # Try to load processed data if exists
        data = pd.read_csv('data/processed/processed_data.csv')
    except:
        # Create sample data if file doesn't exist
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'City': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'], n_samples),
            'PM2.5': np.random.uniform(20, 200, n_samples),
            'PM10': np.random.uniform(30, 300, n_samples),
            'NO2': np.random.uniform(10, 100, n_samples),
            'SO2': np.random.uniform(5, 80, n_samples),
            'CO': np.random.uniform(0.5, 15, n_samples),
            'O3': np.random.uniform(10, 150, n_samples),
            'Month': np.random.randint(1, 13, n_samples),
            'Season': np.random.randint(0, 4, n_samples)
        })
        
        # Calculate AQI (simplified)
        data['AQI'] = (data['PM2.5'] * 2 + data['PM10'] * 0.5 + 
                       data['NO2'] * 0.8 + data['SO2'] * 0.3 + 
                       data['CO'] * 5 + data['O3'] * 0.5).clip(0, 500)
    
    return data

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load trained model"""
    try:
        predictor = AQIPredictor('models/best_model.pkl')
        return predictor
    except:
        st.warning("Model not found. Please train the model first using run.py")
        return None

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🌍 Air Quality Index (AQI) Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown("### Analyze, Predict, and Monitor Air Quality in Indian Cities")
    
    # Load data
    data = load_sample_data()
    predictor = load_predictor()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["📊 Data Analysis", "🤖 AQI Prediction", "📈 Pollution Insights", "ℹ️ About"]
    )
    
    # Page routing
    if page == "📊 Data Analysis":
        show_data_analysis(data)
    elif page == "🤖 AQI Prediction":
        show_prediction_page(predictor, data)
    elif page == "📈 Pollution Insights":
        show_insights_page(data)
    else:
        show_about_page()

def show_data_analysis(data):
    """Display data analysis page"""
    st.markdown('<div class="sub-header">Data Overview</div>', unsafe_allow_html=True)
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Cities", data['City'].nunique() if 'City' in data.columns else 'N/A')
    with col3:
        st.metric("Avg AQI", f"{data['AQI'].mean():.1f}")
    with col4:
        st.metric("Max AQI", f"{data['AQI'].max():.1f}")
    
    # Data preview
    st.markdown("#### Sample Data")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Pollutant distribution
    st.markdown("#### Pollutant Concentration Distribution")
    
    pollutants_available = [col for col in POLLUTANT_COLUMNS if col in data.columns]
    
    if pollutants_available:
        selected_pollutant = st.selectbox("Select Pollutant:", pollutants_available)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                data, x=selected_pollutant,
                nbins=50,
                title=f"{selected_pollutant} Distribution",
                labels={selected_pollutant: f"{selected_pollutant} (µg/m³)"},
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                data, y=selected_pollutant,
                title=f"{selected_pollutant} Box Plot",
                labels={selected_pollutant: f"{selected_pollutant} (µg/m³)"},
                color_discrete_sequence=['#ff7f0e']
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    # AQI distribution by city
    if 'City' in data.columns:
        st.markdown("#### AQI Distribution by City")
        
        fig_city = px.box(
            data, x='City', y='AQI',
            title="AQI Variation Across Cities",
            labels={'AQI': 'Air Quality Index', 'City': 'City'},
            color='City'
        )
        st.plotly_chart(fig_city, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("#### Pollutant Correlation Matrix")
    
    pollutants_for_corr = [col for col in POLLUTANT_COLUMNS + ['AQI'] if col in data.columns]
    if pollutants_for_corr:
        corr_matrix = data[pollutants_for_corr].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Correlation Between Pollutants and AQI',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def show_prediction_page(predictor, data):
    """Display prediction page"""
    st.markdown('<div class="sub-header">Predict AQI</div>', unsafe_allow_html=True)
    
    if predictor is None:
        st.error("Model not loaded. Please train the model first.")
        return
    
    st.markdown("#### Enter Pollutant Concentrations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0, 1.0)
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0, 1.0)
        no2 = st.slider("NO₂ (µg/m³)", 0.0, 200.0, 40.0, 1.0)
    
    with col2:
        so2 = st.slider("SO₂ (µg/m³)", 0.0, 150.0, 20.0, 1.0)
        co = st.slider("CO (mg/m³)", 0.0, 30.0, 2.0, 0.1)
        o3 = st.slider("O₃ (µg/m³)", 0.0, 200.0, 50.0, 1.0)
    
    # Additional features
    col3, col4 = st.columns(2)
    with col3:
        month = st.selectbox("Month", list(range(1, 13)), index=5)
    with col4:
        season = st.selectbox("Season", 
                             ["Winter (0)", "Spring (1)", "Summer (2)", "Autumn (3)"],
                             index=2)
        season_value = int(season.split("(")[1].split(")")[0])
    
    # Predict button
    if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
        # Prepare features
        features = {
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'SO2': so2,
            'CO': co,
            'O3': o3,
            'Month': month,
            'Season': season_value
        }
        
        # Get prediction
        result = predictor.predict_with_details(features)
        
        # Display result
        st.markdown("---")
        st.markdown("#### Prediction Result")
        
        # AQI display
        aqi_html = f"""
        <div class="aqi-display" style="background-color: {result['color']}; color: white;">
            AQI: {result['aqi']}<br>
            <span style="font-size: 1.5rem;">{result['category']}</span>
        </div>
        """
        st.markdown(aqi_html, unsafe_allow_html=True)
        
        # Health message
        st.info(result['health_message'])
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.markdown(f"""
            <div class="recommendation-box">
                {i}. {rec}
            </div>
            """, unsafe_allow_html=True)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['aqi'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AQI Level"},
            gauge={
                'axis': {'range': [None, 500]},
                'bar': {'color': result['color']},
                'steps': [
                    {'range': [0, 50], 'color': AQI_COLORS['Good']},
                    {'range': [51, 100], 'color': AQI_COLORS['Satisfactory']},
                    {'range': [101, 200], 'color': AQI_COLORS['Moderate']},
                    {'range': [201, 300], 'color': AQI_COLORS['Poor']},
                    {'range': [301, 400], 'color': AQI_COLORS['Very Poor']},
                    {'range': [401, 500], 'color': AQI_COLORS['Severe']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': result['aqi']
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

def show_insights_page(data):
    """Display pollution insights page"""
    st.markdown('<div class="sub-header">Pollution Insights</div>', unsafe_allow_html=True)
    
    # Seasonal trends
    if 'Month' in data.columns and 'AQI' in data.columns:
        st.markdown("#### Seasonal AQI Trends")
        
        monthly_aqi = data.groupby('Month')['AQI'].mean().reset_index()
        
        fig_seasonal = px.line(
            monthly_aqi, x='Month', y='AQI',
            title='Average AQI by Month',
            markers=True,
            labels={'Month': 'Month', 'AQI': 'Average AQI'}
        )
        fig_seasonal.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Dominant pollutants
    st.markdown("#### Major Pollution Contributors")
    
    pollutants_available = [col for col in POLLUTANT_COLUMNS if col in data.columns]
    
    if pollutants_available:
        pollutant_avg = data[pollutants_available].mean().sort_values(ascending=False)
        
        fig_contributors = px.bar(
            x=pollutant_avg.values,
            y=pollutant_avg.index,
            orientation='h',
            title='Average Pollutant Concentrations',
            labels={'x': 'Average Concentration (µg/m³)', 'y': 'Pollutant'},
            color=pollutant_avg.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_contributors, use_container_width=True)
    
    # AQI category distribution
    if 'AQI' in data.columns:
        st.markdown("#### AQI Category Distribution")
        
        def categorize(aqi):
            for cat, (low, high) in AQI_BREAKPOINTS.items():
                if low <= aqi <= high:
                    return cat
            return 'Severe'
        
        data['AQI_Category'] = data['AQI'].apply(categorize)
        category_counts = data['AQI_Category'].value_counts()
        
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Distribution of AQI Categories',
            color=category_counts.index,
            color_discrete_map=AQI_COLORS
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # City-wise risk analysis
    if 'City' in data.columns and 'AQI' in data.columns:
        st.markdown("#### City-wise Air Quality Risk")
        
        city_stats = data.groupby('City')['AQI'].agg(['mean', 'max', 'min']).round(1)
        city_stats = city_stats.sort_values('mean', ascending=False)
        
        st.dataframe(city_stats, use_container_width=True)

def show_about_page():
    """Display about page"""
    st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This Air Quality Index (AQI) Prediction System is a comprehensive data science application
    designed to analyze, predict, and monitor air quality in Indian cities.
    
    ### 🔧 Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python, Pandas, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Visualization**: Plotly, Matplotlib, Seaborn
    
    ### 📊 Features
    
    1. **Data Analysis**: Comprehensive EDA with interactive visualizations
    2. **AQI Prediction**: Real-time prediction using trained ML models
    3. **Pollution Insights**: Identify major contributors and trends
    4. **Health Recommendations**: Actionable advice based on AQI levels
    
    ### 🤖 Machine Learning Pipeline
    
    1. Data loading and validation
    2. Preprocessing (missing values, outliers, scaling)
    3. Feature engineering (temporal features, interactions)
    4. Model training (Linear Regression, Random Forest, XGBoost)
    5. Model evaluation and selection
    6. Deployment and inference
    
    ### 📈 AQI Categories (Indian Standards)
    
    - **Good (0-50)**: Minimal impact
    - **Satisfactory (51-100)**: Minor breathing discomfort to sensitive people
    - **Moderate (101-200)**: Breathing discomfort to people with lungs, asthma and heart diseases
    - **Poor (201-300)**: Breathing discomfort to most people on prolonged exposure
    - **Very Poor (301-400)**: Respiratory illness on prolonged exposure
    - **Severe (401-500)**: Affects healthy people and seriously impacts those with existing diseases
    
    ### 👨‍💻 Developer
    
    Built as an internship-grade project demonstrating:
    - Clean code architecture
    - ML best practices
    - Professional UI/UX design
    - Production-ready implementation
    
    ### 📧 Contact
    
    For queries or feedback, please reach out through the project repository.
    """)

if __name__ == "__main__":
    main()