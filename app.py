import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Grade Impact Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for a Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f2;
    }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00b09b, #96c93d);
        color: white;
        border: none;
        padding: 15px 32px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .result-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 25px;
        border: 2px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the new 7-feature model
@st.cache_resource
def load_model():
    return joblib.load('model1.pkl')

model = load_model()

# App Header
st.title("🎓 AI Impact Analytics")
st.markdown("Discover how your AI usage patterns correlate with academic outcomes.")
st.divider()

# Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=60, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate", "PhD"])
        city = st.text_input("Current City", "New York")

    with col2:
        ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Gemini", "Claude", "Other"])
        usage_hours = st.slider("Daily Usage (Hours)", 0.0, 24.0, 2.5)
        purpose = st.selectbox("Primary Usage Purpose", ["Research", "Coding", "Writing", "General Inquiry"])

# Mapping Dictionary (Ensure these match your training encoding)
mapping = {
    "Male": 0, "Female": 1, "Other": 2,
    "High School": 0, "Undergraduate": 1, "Postgraduate": 2, "PhD": 3,
    "ChatGPT": 0, "Gemini": 1, "Claude": 2, "Other": 3,
    "Research": 0, "Coding": 1, "Writing": 2, "General Inquiry": 3
}

if st.button("Analyze Impact"):
    try:
        # Construct DataFrame with exactly 7 features in correct order
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [mapping.get(gender, 0)],
            'Education_Level': [mapping.get(education, 0)],
            'City': [0], # City usually requires LabelEncoding; using 0 as fallback
            'AI_Tool_Used': [mapping.get(ai_tool, 0)],
            'Daily_Usage_Hours': [usage_hours],
            'Purpose': [mapping.get(purpose, 0)]
        })

        # Make Prediction
        prediction = model.predict(input_data)[0]
        
        # UI Response Styling
        if prediction == "High":
            bg_color, text_color, icon = "#e8f5e9", "#2e7d32", "🌟"
        elif prediction == "Medium":
            bg_color, text_color, icon = "#fff8e1", "#f57f17", "📈"
        else:
            bg_color, text_color, icon = "#ffebee", "#c62828", "⚠️"

        st.markdown(f"""
            <div class="result-card" style="background-color: {bg_color}; color: {text_color};">
                <h2 style="margin: 0;">{icon} Predicted Impact: {prediction}</h2>
                <p style="font-size: 16px; opacity: 0.8;">Based on 7-feature K-Neighbors Analysis</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Analysis Error: {e}")

st.sidebar.caption("Model Version: 1.8.0 | Neighbors: 5")
