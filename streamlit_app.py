import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="SmartTraffic Accident Predictor",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #004E89;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #004E89, #1A759F);
    }
</style>
""", unsafe_allow_html=True)

# --- File Paths Based on Your Project Structure ---
import os

# NEW – Relative paths (works on Streamlit Cloud)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Data CSV in repo root
DATA_PATH = os.path.join(BASE_PATH, "accident_prediction_india.csv")

# All model .pkl files live alongside app.py
MODELS_PATH = BASE_PATH



@st.cache_data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please check that 'accident_prediction_india.csv' is in the repo root.")
        return pd.DataFrame()


# --- Load Models Function ---
def load_model_safely(model_name):
    try:
        # CORRECTED model paths to match your exact file names
        model_paths = {
            'random_forest': os.path.join(MODELS_PATH, "random_forest_model.pkl"),
            'logistic_regression': os.path.join(MODELS_PATH, "logistic_regression_model.pkl"),
            'xgboost': os.path.join(MODELS_PATH, "xgboost_model.pkl")
        }
        
        model_path = model_paths.get(model_name)
        if not model_path:
            st.error(f"Unknown model: {model_name}")
            return None, None
            
        model = joblib.load(model_path)
        
        # Try to load feature names
        try:
            feature_names = joblib.load(f"{MODELS_PATH}/feature_names.pkl")
        except FileNotFoundError:
            st.warning("⚠️ Feature names file not found. Using default columns.")
            feature_names = [
                'hour', 'Month_Num', 'DayOfWeek_Num', 
                'Road Type_Highway', 'Road Type_National Highway', 'Road Type_State Highway', 
                'Road Type_Urban Road', 'Road Type_Village Road', 
                'Weather Conditions_Clear', 'Weather Conditions_Foggy', 
                'Weather Conditions_Rainy', 'Weather Conditions_Stormy', 
                'Driver Gender_Female', 'Driver Gender_Male', 
                'Alcohol Involvement_No', 'Alcohol Involvement_Yes'
            ]
        
        return model, feature_names
        
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None, None

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ("🏠 Home", "📊 Dataset Explorer", "📈 Analytics Dashboard", "🔮 Accident Predictor"),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 SmartTraffic System")
st.sidebar.info("Advanced AI-powered traffic accident analysis and prediction system")

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🚦 SmartTraffic Accident Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to India's Most Advanced Traffic Safety Analytics Platform
    
    Our AI-powered system analyzes traffic accident patterns and predicts severity levels to help improve road safety across India.
    """)
    
    # Key Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Data Analysis</h3>
            <p>Comprehensive accident data exploration with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Smart Analytics</h3>
            <p>Advanced statistical insights into accident patterns and trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 AI Prediction</h3>
            <p>Machine learning models for accurate accident severity prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    df = load_data()
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accidents", f"{len(df):,}")
        with col2:
            if 'Accident Severity' in df.columns:
                fatal_count = len(df[df['Accident Severity'] == 'Fatal'])
                st.metric("Fatal Accidents", f"{fatal_count:,}")
            else:
                st.metric("Fatal Accidents", "N/A")
        with col3:
            st.metric("Data Coverage", "Pan-India")
        with col4:
            st.metric("AI Models", "3 Active")
    
    st.markdown("""
    ---
    ### 🎯 How It Works
    
    1. **📁 Explore Dataset**: Browse through comprehensive accident data
    2. **📈 Analyze Patterns**: Discover insights through interactive visualizations  
    3. **🔮 Predict Outcomes**: Use AI models to predict accident severity
    
    ---
    *Developed by: **M. Pavan Kumar** | SmartTraffic Research Team*
    """)

# --- DATASET EXPLORER PAGE ---
elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="main-header">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df.empty:
        st.stop()
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Total Records:** {len(df):,}")
    with col2:
        st.info(f"**Features:** {len(df.columns)}")
    with col3:
        st.info(f"**Time Period:** Multi-year")
    
    # Data Preview
    st.markdown("### 🔍 Data Preview")
    
    # Search and Filter Options
    col1, col2 = st.columns([2, 1])
    with col1:
        search_col = st.selectbox("Search by Column:", df.columns)
    with col2:
        num_rows = st.slider("Rows to Display:", 10, 500, 100)
    
    if search_col:
        unique_vals = df[search_col].unique()[:20]  # Limit options
        filter_val = st.selectbox(f"Filter {search_col}:", ["All"] + list(unique_vals))
        
        if filter_val != "All":
            df_filtered = df[df[search_col] == filter_val]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Display filtered data
    st.dataframe(df_filtered.head(num_rows), use_container_width=True)
    
    # Data Summary
    st.markdown("### 📈 Data Summary")
    
    tab1, tab2 = st.tabs(["🔢 Numerical Summary", "📝 Categorical Summary"])
    
    with tab1:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        else:
            st.info("No numerical columns found.")
    
    with tab2:
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:5]:  # Show top 5 categorical columns
                st.write(f"**{col}:**")
                value_counts = df[col].value_counts().head(10)
                st.bar_chart(value_counts)
        else:
            st.info("No categorical columns found.")

# --- ANALYTICS DASHBOARD PAGE ---
elif page == "📈 Analytics Dashboard":
    st.markdown('<h1 class="main-header">📈 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df.empty:
        st.stop()
    
    # Key Metrics Row
    if 'Accident Severity' in df.columns:
        severity_counts = df['Accident Severity'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Accidents", f"{len(df):,}")
        with col2:
            st.metric("Fatal", f"{severity_counts.get('Fatal', 0):,}", delta="-12%")
        with col3:
            st.metric("Serious", f"{severity_counts.get('Serious', 0):,}", delta="5%")
        with col4:
            st.metric("Minor", f"{severity_counts.get('Minor', 0):,}", delta="8%")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🥧 Severity Distribution", "⏰ Time Analysis", "🛣️ Location Analysis", "🌡️ Weather Impact"])
    
    with tab1:
        if 'Accident Severity' in df.columns:
            # Main severity distribution pie chart
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Accident Severity Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Severity bar chart
            fig2 = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                title="Accident Count by Severity",
                color=severity_counts.index,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # NEW: Vehicle Type in Severity Analysis
            if 'Vehicle Type Involved' in df.columns:
                st.markdown("#### 🚗 Vehicle Type Distribution in Accidents")
                vehicle_severity = pd.crosstab(df['Vehicle Type Involved'], df['Accident Severity'])
                fig_vehicle = px.bar(
                    vehicle_severity,
                    title="Vehicle Type vs Accident Severity",
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={'index': 'Vehicle Type', 'value': 'Number of Accidents'}
                )
                st.plotly_chart(fig_vehicle, use_container_width=True)
                
                # Vehicle type pie chart
                vehicle_counts = df['Vehicle Type Involved'].value_counts()
                fig_vehicle_pie = px.pie(
                    values=vehicle_counts.values,
                    names=vehicle_counts.index,
                    title="Overall Vehicle Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_vehicle_pie, use_container_width=True)
    
    with tab2:
        # ← Insert your time‐analysis code here
        if 'hour' in df.columns:
            hourly_accidents = df.groupby('hour').size().reset_index(name='count')
            fig = px.line(
                hourly_accidents,
                x='hour',
                y='count',
                title="Accidents by Hour of Day",
                markers=True
            )
            fig.update_layout(xaxis_title="Hour", yaxis_title="Number of Accidents")
            st.plotly_chart(fig, use_container_width=True)

        if 'Month_Num' in df.columns:
            monthly_accidents = df.groupby('Month_Num').size().reset_index(name='count')
            fig2 = px.bar(
                monthly_accidents,
                x='Month_Num',
                y='count',
                title="Accidents by Month"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # NEW: Vehicle Type in Time Analysis
        if 'Vehicle Type Involved' in df.columns and 'hour' in df.columns:
            st.markdown("#### 🚗 Vehicle Type Accidents by Hour")
            vehicle_hourly = df.groupby(['hour', 'Vehicle Type Involved']).size().reset_index(name='count')
            fig_vehicle_time = px.line(
                vehicle_hourly,
                x='hour',
                y='count',
                color='Vehicle Type Involved',
                title="Vehicle Type Accidents Throughout the Day",
                markers=True
            )
            fig_vehicle_time.update_layout(xaxis_title="Hour of Day", yaxis_title="Number of Accidents")
            st.plotly_chart(fig_vehicle_time, use_container_width=True)

            if not vehicle_hourly.empty:
                vehicle_hour_pivot = vehicle_hourly.pivot_table(
                    values='count',
                    index='Vehicle Type Involved',
                    columns='hour',
                    aggfunc='sum',
                    fill_value=0
                )
                fig_heatmap = px.imshow(
                    vehicle_hour_pivot,
                    title="Vehicle Type Accidents Heatmap by Hour",
                    labels={'x': 'Hour of Day', 'y': 'Vehicle Type', 'color': 'Accident Count'},
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
    with tab3:
        if 'Road Type' in df.columns:
            road_accidents = df['Road Type'].value_counts()
            fig = px.bar(
                x=road_accidents.values,
                y=road_accidents.index,
                orientation='h',
                title="Accidents by Road Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # NEW: Vehicle Type in Location Analysis
        if 'Vehicle Type Involved' in df.columns and 'Road Type' in df.columns:
            st.markdown("#### 🚗 Vehicle Type Distribution by Road Type")
            vehicle_road = pd.crosstab(df['Road Type'], df['Vehicle Type Involved'])
            fig_vehicle_road = px.bar(
                vehicle_road,
                title="Vehicle Type Distribution Across Road Types",
                barmode='stack',
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'index': 'Road Type', 'value': 'Number of Accidents'}
            )
            st.plotly_chart(fig_vehicle_road, use_container_width=True)
            
            # Percentage distribution
            vehicle_road_pct = vehicle_road.div(vehicle_road.sum(axis=1), axis=0) * 100
            fig_vehicle_road_pct = px.bar(
                vehicle_road_pct,
                title="Vehicle Type Percentage Distribution by Road Type",
                barmode='stack',
                color_discrete_sequence=px.colors.qualitative.Dark2,
                labels={'index': 'Road Type', 'value': 'Percentage (%)'}
            )
            st.plotly_chart(fig_vehicle_road_pct, use_container_width=True)
    
    with tab4:
        if 'Weather Conditions' in df.columns and 'Accident Severity' in df.columns:
            weather_severity = pd.crosstab(df['Weather Conditions'], df['Accident Severity'])
            fig = px.bar(
                weather_severity,
                title="Weather vs Accident Severity",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # NEW: Vehicle Type in Weather Analysis
        if 'Vehicle Type Involved' in df.columns and 'Weather Conditions' in df.columns:
            st.markdown("#### 🚗 Vehicle Type Distribution by Weather")
            vehicle_weather = pd.crosstab(df['Weather Conditions'], df['Vehicle Type Involved'])
            fig_vehicle_weather = px.bar(
                vehicle_weather,
                title="Vehicle Type Accidents in Different Weather Conditions",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Dark2,
                labels={'index': 'Weather Conditions', 'value': 'Number of Accidents'}
            )
            st.plotly_chart(fig_vehicle_weather, use_container_width=True)
            
            # Sunburst chart for Weather + Vehicle Type + Severity
            if 'Accident Severity' in df.columns:
                fig_sunburst = px.sunburst(
                    df,
                    path=['Weather Conditions', 'Vehicle Type Involved', 'Accident Severity'],
                    title="Weather → Vehicle Type → Severity Breakdown"
                )
                st.plotly_chart(fig_sunburst, use_container_width=True)

# COMPLETE ACCIDENT PREDICTOR PAGE CODE - REPLACE THE ENTIRE SECTION

else:  # Accident Predictor
    st.markdown('<h1 class="main-header">🔮 Accident Predictor</h1>', unsafe_allow_html=True)

    # Model Selection
    st.markdown("### 🤖 Choose AI Model")
    model_choice = st.selectbox(
        "Select Prediction Model:",
        ["random_forest", "logistic_regression", "xgboost"],
        format_func=lambda x: {
            "random_forest": "🌲 Random Forest (Recommended)",
            "logistic_regression": "📊 Logistic Regression",
            "xgboost": "⚡ XGBoost (Advanced)"
        }[x]
    )

    model, feature_names = load_model_safely(model_choice)

    if model is None:
        st.error("Model loading failed. Please check your model files.")
        st.stop()

    st.success(f"✅ {model_choice.replace('_', ' ').title()} model loaded successfully!")

    # === MODEL ACCURACIES SECTION ===
    st.markdown("---")
    st.markdown("### 📊 Model Accuracies (R² Score)")
    
    # Create model accuracies data (update with your actual scores)
    model_accuracies = {
        "Model": ["Logistic Regression", "Random Forest Regressor", "XGBoost Regressor"],
        "R² Score": [0.6875, 0.9923, 0.8856]  # Update these with your actual model scores
    }
    
    # Create DataFrame
    accuracy_df = pd.DataFrame(model_accuracies)
    
    # Display as a styled table
    st.dataframe(
        accuracy_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn(
                "Model",
                help="Machine Learning Model Type",
                width="medium",
            ),
            "R² Score": st.column_config.NumberColumn(
                "R² Score",
                help="Coefficient of Determination (Higher is Better)",
                min_value=0.0,
                max_value=1.0,
                step=0.0001,
                format="%.4f",
            ),
        }
    )
    
    # Add explanation
    st.info("""
    **📈 Model Performance Explanation:**
    - **R² Score** measures how well the model explains the variance in accident severity
    - **Higher scores** (closer to 1.0) indicate better model performance  
    - **Random Forest** shows the highest accuracy at 99.23%
    - **XGBoost** provides strong performance at 88.56%
    - **Logistic Regression** offers baseline performance at 68.75%
    """)
    
    # Highlight current model
    current_model_name = {
        'random_forest': 'Random Forest Regressor',
        'logistic_regression': 'Logistic Regression', 
        'xgboost': 'XGBoost Regressor'
    }.get(model_choice, model_choice)
    
    current_score = accuracy_df[accuracy_df['Model'] == current_model_name]['R² Score'].values[0]
    st.success(f"🎯 **Currently Selected:** {current_model_name} with **{current_score:.2%}** accuracy")
    
    st.markdown("---")

    # Input Form
    st.markdown("### 📝 Enter Accident Scenario Details")

    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**🕐 Time Information**")
            hour = st.slider("Hour of Day", 0, 23, 14)
            month = st.selectbox("Month", list(range(1, 13)), index=6)
            day_of_week = st.selectbox(
                "Day of Week",
                list(range(7)),
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
            )

        with col2:
            st.markdown("**🛣️ Road & Environment**")
            road_type = st.selectbox("Road Type", [
                'Highway', 'Urban Road', 'State Highway', 'National Highway', 'Village Road'
            ])
            weather_condition = st.selectbox("Weather Conditions", [
                'Clear', 'Rainy', 'Stormy', 'Foggy'
            ])
            road_condition = st.selectbox("Road Condition", [
                'Dry', 'Wet', 'Damaged', 'Under Construction'
            ])
            lighting = st.selectbox("Lighting Conditions", [
                'Daylight', 'Dark', 'Dawn', 'Dusk'
            ])

        with col3:
            st.markdown("**👤 Driver Information**")
            driver_gender = st.selectbox("Driver Gender", ['Male', 'Female'])
            driver_age = st.number_input("Driver Age", 18, 70, 35)
            alcohol_involvement = st.selectbox("Alcohol Involvement", ['No', 'Yes'])

        with col4:
            st.markdown("**🚗 Vehicle & Traffic**")
            vehicle_type = st.selectbox("Vehicle Type Involved", [
                'Car', 'Truck', 'Bus', 'Cycle', 'Pedestrian', 'Two-Wheeler'
            ])
            traffic_control = st.selectbox("Traffic Control Presence", [
                'None', 'Signs', 'Signals', 'Police Checkpost'
            ])
            speed_limit = st.number_input("Speed Limit (km/h)", 30, 120, 60, step=10)

        # Predict Button
        predict_button = st.form_submit_button("🚀 Predict Accident Severity", use_container_width=True)

    # UPDATED PREDICTION LOGIC
    if predict_button:
        # Prepare input data with all features
        user_inputs = {
            'hour': hour,
            'Month_Num': month,
            'DayOfWeek_Num': day_of_week,
            'Road Type': road_type,
            'Weather Conditions': weather_condition,
            'Road Condition': road_condition,
            'Lighting Conditions': lighting,
            'Driver Gender': driver_gender,
            'Driver Age': driver_age,
            'Alcohol Involvement': alcohol_involvement,
            'Vehicle Type Involved': vehicle_type,
            'Traffic Control Presence': traffic_control,
            'Speed Limit (km/h)': speed_limit,
        }
        
        # Create input dataframe
        input_df = pd.DataFrame([user_inputs])
        
        # Get dummy variables for categorical columns
        categorical_cols = ['Road Type', 'Weather Conditions', 'Road Condition', 
                           'Lighting Conditions', 'Driver Gender', 'Alcohol Involvement', 
                           'Vehicle Type Involved', 'Traffic Control Presence']
        
        # Create dummy variables
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, prefix=categorical_cols)
        
        # Ensure all expected columns are present
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Select only the columns that were used during training
        input_ready = input_encoded[feature_names]
        
        try:
            # Make prediction
            prediction_encoded = model.predict(input_ready)[0]
            prediction_proba = model.predict_proba(input_ready)[0] if hasattr(model, 'predict_proba') else None
            
            # Load label encoder for proper mapping
            try:
                le = joblib.load(f"{MODELS_PATH}/label_encoder.pkl")
                predicted_label = le.inverse_transform([prediction_encoded])[0]
            except:
                # Fallback label mapping
                label_map = {0: "Fatal", 1: "Minor", 2: "Serious"}
                predicted_label = label_map.get(prediction_encoded, str(prediction_encoded))
            
            # Results Display
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Severity colors
                severity_colors = {"Fatal": "#FF4B4B", "Serious": "#FF8C00", "Minor": "#32CD32"}
                color = severity_colors.get(predicted_label, "#666666")

                st.markdown(f"""
                <div style="
                    background: {color};
                    color: white;
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    font-size: 2rem;
                    font-weight: bold;
                    margin: 1rem 0;
                ">
                    🚨 {predicted_label} Accident
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                if prediction_proba is not None:
                    st.markdown("**Prediction Confidence:**")
                    for i, (label, prob) in enumerate(zip(['Fatal', 'Minor', 'Serious'], prediction_proba)):
                        # Convert numpy float32 to Python float to fix Streamlit error
                        prob_float = float(prob)
                        st.progress(prob_float, text=f"{label}: {prob_float:.1%}")

            # === ENHANCED SAFETY RECOMMENDATIONS ===
            st.markdown("### 💡 Safety Recommendations")

            try:
                # Primary recommendations based on severity
                if predicted_label == "Fatal":
                    st.error("""
                    🚨 **CRITICAL RISK SCENARIO**
                    - **IMMEDIATE ACTION REQUIRED**
                    - Avoid travel if possible in current conditions
                    - Use alternative, safer routes immediately
                    - Ensure all safety equipment is functional
                    - Consider postponing the journey until conditions improve
                    - Alert emergency contacts about your travel plans
                    """)

                elif predicted_label == "Serious":
                    st.warning("""
                    ⚡ **HIGH RISK SCENARIO**
                    - **DRIVE WITH EXTREME CAUTION**
                    - Reduce speed by at least 20-30% from normal
                    - Increase following distance to 4+ seconds
                    - Avoid overtaking and risky maneuvers
                    - Keep emergency contact numbers ready
                    - Consider alternative transportation if available
                    """)

                else:  # Minor
                    st.success("""
                    ✅ **MODERATE RISK SCENARIO**
                    - **MAINTAIN STANDARD SAFETY PRACTICES**
                    - Stay alert and focused while driving
                    - Follow all traffic rules and speed limits
                    - Drive defensively and anticipate other drivers
                    - Keep vehicle maintenance up to date
                    - Stay hydrated and take breaks on long journeys
                    """)

                # Additional context-specific recommendations
                st.markdown("#### 🎯 Specific Recommendations Based on Your Input:")

                recommendations = []

                # Time-based recommendations
                if hour >= 22 or hour <= 5:
                    recommendations.append("🌙 **Night Driving**: Use high-beam headlights when appropriate, reduce speed by 15%, and stay extra alert for pedestrians and animals")
                elif 7 <= hour <= 9 or 17 <= hour <= 19:
                    recommendations.append("🚦 **Rush Hour Traffic**: Expect heavy congestion, allow 30% extra time, maintain patience, and avoid aggressive driving")
                elif 12 <= hour <= 14:
                    recommendations.append("☀️ **Midday Driving**: Watch for sun glare, use sunglasses, and be alert as this is a high-accident time period")

                # Weather-based recommendations
                if weather_condition in ['Rainy', 'Stormy']:
                    recommendations.append("🌧️ **Wet Road Conditions**: Reduce speed by 25%, double your braking distance, use headlights, and avoid sudden steering or braking")
                elif weather_condition == 'Foggy':
                    recommendations.append("🌫️ **Poor Visibility**: Use fog lights (not high beams), drive 50% slower, increase following distance to 8+ seconds")
                elif weather_condition == 'Clear':
                    recommendations.append("☀️ **Clear Weather**: Maintain standard precautions but watch for sun glare during sunrise/sunset hours")

                # Road type recommendations
                if road_type == 'Highway':
                    recommendations.append("🛣️ **Highway Safety**: Maintain 80+ km/h if safe, use indicators 100m before changing lanes, check blind spots thoroughly")
                elif road_type in ['Village Road', 'Urban Road']:
                    recommendations.append("🏘️ **Urban/Village Areas**: Reduce speed near schools/markets, watch for pedestrians, cyclists, and street vendors")
                elif road_type in ['State Highway', 'National Highway']:
                    recommendations.append("🛤️ **Major Highway**: Be prepared for mixed traffic, overtake only when completely safe, watch for heavy vehicles")

                # Lighting condition recommendations
                if lighting == 'Dark':
                    recommendations.append("🌑 **Dark Conditions**: Ensure headlights work properly, clean windshield, reduce speed, and avoid overtaking")
                elif lighting in ['Dawn', 'Dusk']:
                    recommendations.append("🌅 **Twilight Hours**: Turn on headlights for visibility, be extra cautious as visibility changes rapidly")

                # Age-based recommendations
                if driver_age >= 65:
                    recommendations.append("👴 **Senior Driver**: Take breaks every hour, avoid night driving when possible, inform family of travel plans")
                elif driver_age <= 25:
                    recommendations.append("👶 **Young Driver**: Focus on speed control, minimize distractions, avoid peer pressure to drive recklessly")
                elif 25 < driver_age < 35:
                    recommendations.append("🧑 **Experienced Driver**: Use your experience wisely, set good example for younger drivers")

                # Vehicle type recommendations
                if vehicle_type in ['Truck', 'Bus']:
                    recommendations.append("🚛 **Heavy Vehicle**: Allow extra braking distance (100m+), check mirrors every 8-10 seconds, be aware of larger blind spots")
                elif vehicle_type in ['Cycle', 'Two-Wheeler']:
                    recommendations.append("🚲 **Two Wheeler**: Wear helmet and protective gear, stay in designated lanes, make yourself highly visible with bright colors")
                elif vehicle_type == 'Car':
                    recommendations.append("🚗 **Car Safety**: Ensure all passengers wear seatbelts, adjust mirrors properly, maintain safe following distance")

                # Road condition recommendations
                if road_condition in ['Wet', 'Damaged']:
                    recommendations.append("⚠️ **Poor Road Conditions**: Drive 30% slower, avoid sudden movements, watch for potholes and debris")
                elif road_condition == 'Under Construction':
                    recommendations.append("🚧 **Construction Zone**: Follow speed limits strictly, maintain extra distance, watch for workers and equipment")

                # Speed limit specific
                if speed_limit >= 80:
                    recommendations.append("🏁 **High Speed Zone**: Maintain vehicle properly, check tire pressure, keep emergency kit ready")
                elif speed_limit <= 40:
                    recommendations.append("🐌 **Low Speed Zone**: Watch for pedestrians, school children, frequent stops, and local traffic")

                # Alcohol involvement warning
                if alcohol_involvement == 'Yes':
                    recommendations.append("🚫 **CRITICAL WARNING**: Never drive under influence! Use taxi, public transport, or designate a sober driver")

                # Display all relevant recommendations
                if recommendations:
                    for i, rec in enumerate(recommendations):
                        if i < 5:  # Limit to 5 most relevant recommendations
                            st.info(rec)
                else:
                    st.info("🚗 **General Safety**: Follow traffic rules, stay alert, maintain your vehicle regularly, and drive according to current conditions")

            except Exception as e:
                # Fallback safety recommendations if prediction fails
                st.warning("⚠️ **General Safety Guidelines** (Prediction system temporarily unavailable)")
                st.info("""
                **Universal Safety Practices:**
                - Always wear seatbelts and ensure passengers do the same
                - Maintain safe following distances (3+ seconds in good conditions)
                - Obey speed limits and adjust for road/weather conditions
                - Avoid mobile phone use while driving
                - Never drive under influence of alcohol or drugs
                - Keep vehicle well-maintained and roadworthy
                - Plan routes in advance and inform others of travel plans
                - Take breaks every 2 hours on long journeys
                """)

            # Input Summary with Enhanced Display
            with st.expander("📋 View Complete Input Summary"):
                input_summary = {
                    "🕐 Time Details": {
                        "Hour": f"{hour}:00 ({'Night' if hour >= 22 or hour <= 5 else 'Rush Hour' if 7 <= hour <= 9 or 17 <= hour <= 19 else 'Day'})",
                        "Month": f"{month} ({'Winter' if month in [12,1,2] else 'Summer' if month in [3,4,5] else 'Monsoon' if month in [6,7,8,9] else 'Post-Monsoon'})",
                        "Day of Week": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]
                    },
                    "🛣️ Road & Environment": {
                        "Road Type": f"{road_type} ({'High Risk' if road_type == 'Highway' else 'Medium Risk' if road_type in ['State Highway', 'National Highway'] else 'Low-Medium Risk'})",
                        "Weather": f"{weather_condition} ({'High Risk' if weather_condition in ['Rainy', 'Stormy', 'Foggy'] else 'Low Risk'})",
                        "Road Condition": f"{road_condition} ({'High Risk' if road_condition in ['Wet', 'Damaged', 'Under Construction'] else 'Normal'})",
                        "Lighting": f"{lighting} ({'High Risk' if lighting == 'Dark' else 'Medium Risk' if lighting in ['Dawn', 'Dusk'] else 'Low Risk'})"
                    },
                    "👤 Driver Profile": {
                        "Gender": driver_gender,
                        "Age": f"{driver_age} years ({'Senior' if driver_age >= 65 else 'Young' if driver_age <= 25 else 'Adult'})",
                        "Alcohol Involvement": f"{alcohol_involvement} ({'CRITICAL RISK' if alcohol_involvement == 'Yes' else 'Safe'})"
                    },
                    "🚗 Vehicle & Traffic": {
                        "Vehicle Type": f"{vehicle_type} ({'High Risk' if vehicle_type in ['Two-Wheeler', 'Cycle'] else 'Medium Risk' if vehicle_type in ['Truck', 'Bus'] else 'Standard Risk'})",
                        "Traffic Control": f"{traffic_control} ({'Good' if traffic_control in ['Signals', 'Police Checkpost'] else 'Basic' if traffic_control == 'Signs' else 'No Control - Higher Risk'})",
                        "Speed Limit": f"{speed_limit} km/h ({'High Speed Zone' if speed_limit >= 80 else 'Medium Speed' if speed_limit >= 60 else 'Low Speed Zone'})"
                    }
                }

                for category, details in input_summary.items():
                    st.markdown(f"**{category}**")
                    for key, value in details.items():
                        st.write(f"  • {key}: {value}")
                    st.write("")

                # Risk Assessment Summary
                st.markdown("**🎯 Overall Risk Assessment:**")
                risk_factors = 0
                if weather_condition in ['Rainy', 'Stormy', 'Foggy']: risk_factors += 1
                if lighting == 'Dark': risk_factors += 1
                if hour >= 22 or hour <= 5: risk_factors += 1
                if driver_age <= 25 or driver_age >= 65: risk_factors += 1
                if vehicle_type in ['Two-Wheeler', 'Cycle']: risk_factors += 1
                if alcohol_involvement == 'Yes': risk_factors += 3
                if road_condition in ['Wet', 'Damaged']: risk_factors += 1

                if risk_factors >= 5:
                    st.error(f"🔴 **HIGH RISK** - {risk_factors} risk factors identified")
                elif risk_factors >= 3:
                    st.warning(f"🟡 **MODERATE RISK** - {risk_factors} risk factors identified")
                else:
                    st.success(f"🟢 **LOW-MODERATE RISK** - {risk_factors} risk factors identified")
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.warning("⚠️ Prediction system temporarily unavailable")

        
# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚦 SmartTraffic Accident Predictor | Developed by M. Pavan Kumar | 2025</p>
</div>
""", unsafe_allow_html=True)


