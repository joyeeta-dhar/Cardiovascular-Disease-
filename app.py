import streamlit as st
import pandas as pd
import numpy as np
import os
from model_logic import CardioModel
from eda_visuals import (
    plot_age_distribution,
    plot_bmi_kde,
    plot_correlation_heatmap,
    plot_bp_summary
)

# Page configuration
st.set_page_config(
    page_title="Cardiovasc Insight AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF2E63, #08D9D6);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255, 46, 99, 0.4);
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        background: linear-gradient(to right, #08D9D6, #FF2E63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Model
@st.cache_resource
def load_cardio_model():
    model = CardioModel()
    model.load_model()
    return model

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

cardio_model = load_cardio_model()

# Sidebar
with st.sidebar:
    st.title("Cardiovasc AI Settings")
    st.info("🎯 Notebook Model: `model.joblib` loaded.")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload dataset for Analysis (Optional)", type="csv")
    data = None
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.success("Analysis dataset ready!")

    st.markdown("---")
    menu = st.radio("Navigation", ["Prediction", "Data Analysis", "About Project"])

# Header
st.title("Cardiovasc Insight AI")
st.markdown("Predicting cardiovascular disease risk with advanced machine learning.")

if menu == "Prediction":
    st.header("Cardiovascular Risk Assessment")
    st.write("Enter patient metrics below to analyze risk levels.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
        
    with col2:
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
        ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", value=120)
        ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", value=80)
        
    with col3:
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
        active = st.checkbox("Physical Activity", value=True)

    col4, col5 = st.columns(2)
    with col4:
        smoke = st.checkbox("Smoking", value=False)
    with col5:
        alco = st.checkbox("Alcohol Consumption", value=False)

    if st.button("Analyze Risk"):
        if not cardio_model.is_trained:
            st.error("Please upload the dataset and train the model first in the sidebar!")
        else:
            # Prepare input data
            input_data = {
                'age': age_years * 365.25,
                'gender': gender,
                'height': height,
                'weight': weight,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'cholesterol': cholesterol,
                'gluc': gluc,
                'smoke': 1 if smoke else 0,
                'alco': 1 if alco else 0,
                'active': 1 if active else 0
            }
            
            with st.spinner("Analyzing data..."):
                pred, prob = cardio_model.predict(input_data)
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.subheader("Result")
                if pred == 1:
                    st.error("⚠️ HIGH RISK OF CVD")
                else:
                    st.success("✅ LOW RISK OF CVD")
                
                st.metric(label="Risk Probability", value=f"{prob:.2%}")
            
            with res_col2:
                st.subheader("Assessment Details")
                st.write(f"The model has analyzed the patient metrics including blood pressure category and BMI.")
                st.progress(prob)
                if prob > 0.5:
                    st.warning("Lifestyle changes and medical advice are strongly recommended.")
                else:
                    st.info("Maintaining a healthy lifestyle is recommended.")

elif menu == "Data Analysis":
    st.header("Interactive Exploratory Data Analysis")
    if data is None:
        st.info("Please provide the dataset file (local or upload) in the sidebar to see interactive analysis.")
    else:
        # Apply preprocessing for visualization
        data_processed = cardio_model.preprocess_data(data)
        
        tab1, tab2, tab3 = st.tabs(["Age & BMI", "Blood Pressure", "Correlation"])
        
        with tab1:
            st.plotly_chart(plot_age_distribution(data), use_container_width=True)
            st.plotly_chart(plot_bmi_kde(data_processed), use_container_width=True)
            
        with tab2:
            st.plotly_chart(plot_bp_summary(data_processed), use_container_width=True)
            
        with tab3:
            st.plotly_chart(plot_correlation_heatmap(data_processed), use_container_width=True)

elif menu == "About Project":
    st.header("Project Overview")
    st.markdown("""
    This project is built using a dataset of cardiovascular patients. 
    It leverages a **Random Forest Classifier** with hyperparameter tuning to achieve optimal results.
    
    ### Key Features:
    - **Advanced Feature Engineering**: BMI, Age Groups, BP Category, Pulse Pressure.
    - **Interactive Visualizations**: Real-time analysis of dataset metrics.
    - **Scalable Design**: Can handle dynamic dataset uploads and model retraining.
    
    ### Technologies Used:
    - **Streamlit**: For the interactive web interface.
    - **Scikit-Learn & XGBoost**: For robust machine learning.
    - **Plotly**: For high-quality interactive data visualization.
    """)

st.markdown("---")
st.markdown("Developed with ❤️ for Cardiovasc Health Analysis")
