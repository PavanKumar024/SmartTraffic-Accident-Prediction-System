
import pickle
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_streamlit_models():
    """Load all models and components for Streamlit"""
    try:
        with open(r'C:\SmartTraffic_Project\models\streamlit_model_package.pkl', 'rb') as f:
            package = pickle.load(f)
        return package
    except Exception as e:
        st.error(f"Error loading model package: {e}")
        return None

def preprocess_streamlit_input(hour, month, day_of_week, road_type, weather_conditions, 
                              road_condition, lighting_conditions, driver_gender, 
                              driver_age, alcohol_involvement, vehicle_type, 
                              traffic_control, speed_limit):
    """Preprocess input from Streamlit interface exactly like training data"""

    # Create base features exactly as in training
    data = {
        'hour': hour,
        'Month_Nu': month,
        'DayOfWeek_Num': day_of_week,
        'Road Type': road_type.lower().strip(),
        'Weather Conditions': weather_conditions.lower().strip(),
        'Road Condition': road_condition.lower().strip(),
        'Lighting Conditions': lighting_conditions.lower().strip(),
        'Driver Gender': driver_gender.lower().strip(),
        'Driver Age': driver_age,
        'Alcohol Involvement': alcohol_involvement.lower().strip(),
        'Vehicle Type Involved': vehicle_type.lower().strip(),
        'Traffic Control Presence': traffic_control.lower().strip(),
        'Speed Limit (km/h)': speed_limit
    }

    # Enhanced features (exactly as in training)
    data['Is_Weekend'] = 1 if day_of_week >= 5 else 0
    data['Night_Time'] = 1 if (hour >= 22 or hour <= 5) else 0
    data['Rush_Hour'] = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0

    # Weather severity mapping
    weather_severity_map = {
        'clear': 1, 'sunny': 1, 'partly cloudy': 2, 'cloudy': 2, 'overcast': 2,
        'hazy': 3, 'foggy': 4, 'mist': 4, 'drizzle': 4, 'rainy': 5, 'rain': 5,
        'heavy rain': 6, 'stormy': 7, 'thunderstorm': 7, 'snow': 6, 'sleet': 5
    }
    data['Weather_Severity'] = weather_severity_map.get(weather_conditions.lower().strip(), 2)

    # Driver age categories
    data['Young_Driver'] = 1 if driver_age <= 25 else 0
    data['Senior_Driver'] = 1 if driver_age >= 65 else 0

    # High risk vehicle
    high_risk_vehicles = ['two-wheeler', 'cycle', 'pedestrian', 'motorcycle', 'auto-rickshaw']
    data['High_Risk_Vehicle'] = 1 if vehicle_type.lower().strip() in high_risk_vehicles else 0

    # Risk score calculation (exactly as in training)
    data['Risk_Score'] = (
        data['Night_Time'] * 3 +
        data['Weather_Severity'] * 1.5 +
        data['Young_Driver'] * 2 +
        data['Senior_Driver'] * 2.5 +
        (1 if alcohol_involvement.lower().strip() == 'yes' else 0) * 5 +
        data['High_Risk_Vehicle'] * 3 +
        data['Rush_Hour'] * 1.5
    )

    return pd.DataFrame([data])

def predict_accident_severity(model_name, hour, month, day_of_week, road_type, 
                             weather_conditions, road_condition, lighting_conditions, 
                             driver_gender, driver_age, alcohol_involvement, 
                             vehicle_type, traffic_control, speed_limit):
    """Make prediction using Streamlit inputs"""

    # Load models
    package = load_streamlit_models()
    if package is None:
        return None, None

    # Preprocess input
    input_df = preprocess_streamlit_input(
        hour, month, day_of_week, road_type, weather_conditions, 
        road_condition, lighting_conditions, driver_gender, 
        driver_age, alcohol_involvement, vehicle_type, 
        traffic_control, speed_limit
    )

    # One-hot encode (exactly as in training)
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Ensure all required features are present
    required_features = package['feature_names']
    for feature in required_features:
        if feature not in input_encoded.columns:
            input_encoded[feature] = 0

    # Select only required features in correct order
    input_final = input_encoded[required_features]

    # Get model
    model_key = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    if model_key == 'random_forest':
        model_key = 'Random Forest'
    elif model_key == 'logistic_regression':
        model_key = 'Logistic Regression'
    elif model_key == 'xgboost':
        model_key = 'XGBoost'

    model = package['models'][model_key]

    # Make prediction
    try:
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]

        # Convert back to label
        label_encoder = package['label_encoder']
        predicted_severity = label_encoder.inverse_transform([prediction])[0]

        # Get class probabilities
        class_probs = dict(zip(label_encoder.classes_, probability))

        return predicted_severity, class_probs
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def get_model_performance():
    """Get model performance metrics"""
    package = load_streamlit_models()
    if package:
        return package['model_performance'], package['auc_scores']
    return None, None
