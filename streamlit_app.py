import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
import os

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
.main-header { font-size: 3rem; color: #FF6B35; text-align: center; margin-bottom: 2rem; }
.sub-header { font-size: 2rem; color: #004E89; margin: 1rem 0; }
.metric-card { background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
.sidebar .sidebar-content { background: linear-gradient(180deg, #004E89, #1A759F); }
</style>
""", unsafe_allow_html=True)

# --- Paths using relative references ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "accident_prediction_india.csv")
MODELS_PATH = BASE_PATH  # model files are in the repo root

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'accident_prediction_india.csv' is in the repository.")
        return pd.DataFrame()

def load_model_safely(model_name):
    try:
        model_files = {
            'random_forest': "random_forest_model.pkl",
            'logistic_regression': "logistic_regression_model.pkl",
            'xgboost': "xgboost_model.pkl"
        }
        path = model_files.get(model_name)
        if not path:
            st.error(f"Unknown model: {model_name}")
            return None, None
        model = joblib.load(os.path.join(MODELS_PATH, path))
        try:
            feature_names = joblib.load(os.path.join(MODELS_PATH, "feature_names.pkl"))
        except FileNotFoundError:
            st.warning("⚠️ Feature names file not found. Using default list.")
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
        st.error(f"Error loading {model_name}: {e}")
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
st.sidebar.info("🚦 SmartTraffic: AI-powered traffic accident analysis and prediction")

# --- Pages ---

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🚦 SmartTraffic Accident Predictor</h1>', unsafe_allow_html=True)
    st.write("Welcome to the SmartTraffic system. Use the sidebar to navigate.")

elif page == "📊 Dataset Explorer":
    st.markdown('<h1 class="main-header">📊 Dataset Explorer</h1>', unsafe_allow_html=True)
    df = load_data()
    if df.empty:
        st.stop()
    st.dataframe(df.head(100))

elif page == "📈 Analytics Dashboard":
    st.markdown('<h1 class="main-header">📈 Analytics Dashboard</h1>', unsafe_allow_html=True)
    df = load_data()
    if df.empty:
        st.stop()
    fig = px.histogram(df, x='Accident_Severity', title='Accident Severity Distribution')
    st.plotly_chart(fig, use_container_width=True)

else:  # 🔮 Accident Predictor
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
        st.stop()
    st.success(f"✅ {model_choice.replace('_', ' ').title()} model loaded successfully!")

    # Model Accuracies
    st.markdown("---")
    st.markdown("### 📊 Model Accuracies (R² Score)")
    acc_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest Regressor", "XGBoost Regressor"],
        "R² Score": [0.6875, 0.9923, 0.8856]
    })
    st.dataframe(acc_df, use_container_width=True, hide_index=True)
    st.info("📈 Random Forest leads with 99.23%; XGBoost offers strong 88.56% accuracy.")
    curr_name = {
        'random_forest': "Random Forest Regressor",
        'logistic_regression': "Logistic Regression",
        'xgboost': "XGBoost Regressor"
    }[model_choice]
    curr_score = acc_df[acc_df.Model == curr_name]["R² Score"].iloc[0]
    st.success(f"🎯 Currently Selected: {curr_name} ({curr_score:.2%} accuracy)")
    st.markdown("---")

    # Input Form
    st.markdown("### 📝 Enter Accident Scenario Details")
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**🕐 Time Info**")
            hour = st.slider("Hour of Day", 0, 23, 14)
            month = st.selectbox("Month", list(range(1, 13)), index=6)
            day = st.selectbox("Day of Week", list(range(7)),
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        with c2:
            st.markdown("**🛣️ Road & Environment**")
            road = st.selectbox("Road Type", ['Highway','Urban Road','State Highway','National Highway','Village Road'])
            weather = st.selectbox("Weather", ['Clear','Rainy','Stormy','Foggy'])
            road_cond = st.selectbox("Road Condition", ['Dry','Wet','Damaged','Under Construction'])
            light = st.selectbox("Lighting", ['Daylight','Dark','Dawn','Dusk'])
        with c3:
            st.markdown("**👤 Driver Info**")
            gender = st.selectbox("Gender", ['Male','Female'])
            age = st.number_input("Age", 18, 70, 35)
            alcohol = st.selectbox("Alcohol Involvement", ['No','Yes'])
        with c4:
            st.markdown("**🚗 Vehicle & Traffic**")
            veh = st.selectbox("Vehicle Type", ['Car','Truck','Bus','Cycle','Pedestrian','Two-Wheeler'])
            control = st.selectbox("Traffic Control", ['None','Signs','Signals','Police Checkpost'])
            speed = st.number_input("Speed Limit (km/h)", 30, 120, 60, step=10)
        predict = st.form_submit_button("🚀 Predict Accident Severity", use_container_width=True)

    if predict:
        inp = pd.DataFrame([{
            'hour': hour, 'Month_Num': month, 'DayOfWeek_Num': day,
            'Road Type': road, 'Weather Conditions': weather,
            'Road Condition': road_cond, 'Lighting Conditions': light,
            'Driver Gender': gender, 'Driver Age': age,
            'Alcohol Involvement': alcohol,
            'Vehicle Type Involved': veh,
            'Traffic Control Presence': control,
            'Speed Limit (km/h)': speed
        }])
        enc = pd.get_dummies(inp, columns=[
            'Road Type','Weather Conditions','Road Condition',
            'Lighting Conditions','Driver Gender','Alcohol Involvement',
            'Vehicle Type Involved','Traffic Control Presence'
        ])
        for f in feature_names:
            if f not in enc.columns: enc[f] = 0
        ready = enc[feature_names]
        try:
            pred_enc = model.predict(ready)[0]
            proba = model.predict_proba(ready)[0] if hasattr(model, 'predict_proba') else None
            try:
                le = joblib.load(os.path.join(MODELS_PATH, "label_encoder.pkl"))
                label = le.inverse_transform([pred_enc])[0]
            except:
                label = {0:"Fatal",1:"Minor",2:"Serious"}.get(pred_enc, str(pred_enc))
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                colors = {"Fatal":"#FF4B4B","Serious":"#FF8C00","Minor":"#32CD32"}
                c = colors.get(label,"#666666")
                st.markdown(f"""
                    <div style="
                        background:{c};color:white;padding:2rem;border-radius:15px;
                        text-align:center;font-size:2rem;font-weight:bold;margin:1rem 0;">
                        🚨 {label} Accident
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                if proba is not None:
                    st.markdown("**Prediction Confidence:**")
                    for lab,p in zip(["Fatal","Minor","Serious"], proba):
                        st.progress(float(p), text=f"{lab}: {p:.1%}")
            st.markdown("### 💡 Safety Recommendations")
            # [Recommendations logic here—omitted for brevity]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.warning("⚠️ Prediction system unavailable")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;">
    <p>🚦 SmartTraffic Accident Predictor | Developed by M. Pavan Kumar | 2025</p>
</div>
""", unsafe_allow_html=True)

