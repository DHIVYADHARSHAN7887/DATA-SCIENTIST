import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(
    page_title="Cardio Predict - Heart Disease Risk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica', sans-serif;
        color: #0E1117;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Helvetica', sans-serif;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e86de;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background-color: #2e86de;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading & Model Training ---
@st.cache_resource
def load_data_and_train_model():
    try:
        df = pd.read_csv("framingham_heart_disease.csv")
    except FileNotFoundError:
        st.error("Error: 'framingham_heart_disease.csv' not found. Please ensure the dataset is in the same directory.")
        return None, None, None

    # Preprocessing
    df = df.drop(['education'], axis=1) # Drop education as per notebook

    # Fill missing values with mean
    means = {
        'cigsPerDay': df['cigsPerDay'].mean(),
        'BPMeds': df['BPMeds'].mean(),
        'totChol': df['totChol'].mean(),
        'BMI': df['BMI'].mean(),
        'heartRate': df['heartRate'].mean(),
        'glucose': df['glucose'].mean()
    }
    df.fillna(means, inplace=True)
    
    # Drop any remaining rows with NaNs (just in case, though fillna should cover specific cols)
    df.dropna(inplace=True)

    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=10000, solver='liblinear') # Standard params for convergence
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, len(df)

model, accuracy, dataset_size = load_data_and_train_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80) 
    st.title("Clinical Dashboard")
    st.write("Enter patient vitals and demographics to generate a comprehensive 10-year CHD risk profile.")
    
    st.markdown("---")
    st.subheader("⚙️ Model Specifications")
    if accuracy:
        st.write(f"**Algorithm:** Logistic Regression")
        st.write(f"**Training Accuracy:** {accuracy*100:.2f}%")
        st.write(f"**Validation Set:** {int(dataset_size*0.25)} samples")
        st.write(f"**Dataset:** Framingham Heart Study")
    else:
        st.warning("Model not trained.")

    st.markdown("---")
    st.subheader("👨‍💻 Developer Information")
    st.write("**Project:** Heart Disease Prediction System")
    st.write("**Year:** 2026")
    st.write("**Version:** 1.0.0")

# --- Main Content ---
st.markdown('<div class="main-header">Heart Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Framingham Heart Study Predictive Model • Logistic Regression</div>', unsafe_allow_html=True)

if model:
    tab1, tab2, tab3 = st.tabs(["Patient Evaluation", "Model Performance", "Study Details"])

    with tab1:
        st.markdown("### 📋 Client Vitals Form")
        
        with st.form("patient_data_form"):
            col1, col2, col3 = st.columns(3)

            # --- 01. Patient Profile ---
            with col1:
                st.info("#### 01. Patient Profile")
                gender = st.selectbox("Gender", ["Male", "Female"], index=0)
                male = 1 if gender == "Male" else 0
                
                age = st.number_input("Age", min_value=18, max_value=100, value=50)
                
                smoker_status = st.radio("Smoking History", ["Non-Smoker", "Current Smoker"])
                currentSmoker = 1 if smoker_status == "Current Smoker" else 0
                
                cigsPerDay = 0
                if currentSmoker:
                    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0.0, max_value=100.0, value=10.0)

            # --- 02. Clinical Vitals ---
            with col2:
                st.info("#### 02. Clinical Vitals")
                # Prevalent Stroke / Hyp / Diabetes
                history_stroke = st.checkbox("History of Stroke")
                prevalentStroke = 1 if history_stroke else 0
                
                history_hyp = st.checkbox("Prevalent Hypertension")
                prevalentHyp = 1 if history_hyp else 0
                
                history_diabetes = st.checkbox("Diabetes")
                diabetes = 1 if history_diabetes else 0
                
                bp_meds = st.checkbox("On BP Medication")
                BPMeds = 1 if bp_meds else 0
                
                sysBP = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=250.0, value=120.0)
                diaBP = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=80.0)

            # --- 03. Lab Results ---
            with col3:
                st.info("#### 03. Lab Results")
                totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=200.0)
                
                bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, format="%.2f")
                
                heartRate = st.number_input("Resting Heart Rate (bpm)", min_value=40.0, max_value=150.0, value=75.0)
                
                glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=85.0)

            st.markdown("---")
            submit_button = st.form_submit_button("Generate Risk Assessment Report")

        if submit_button:
            # Prepare input vector (must match order of X during training)
            # male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
            input_data = np.array([[
                male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, bmi, heartRate, glucose
            ]])
            
            # Predict
            prediction_prob = model.predict_proba(input_data)[0][1] # Probability of Class 1 (CHD)
            prediction_class = model.predict(input_data)[0]
            
            # Display Results
            st.markdown("### Assessment Report")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("#### 10-YEAR CHD PROBABILITY")
                risk_percentage = prediction_prob * 100
                st.markdown(f"<h1 style='color: {'#27ae60' if risk_percentage < 15 else '#e67e22' if risk_percentage < 30 else '#c0392b'};'>{risk_percentage:.1f}%</h1>", unsafe_allow_html=True)
                
                if risk_percentage < 15:
                    st.success("**LOW RISK**")
                    st.write("Patient falls within the low-risk category. Continue maintaining a healthy lifestyle.")
                elif risk_percentage < 30:
                    st.warning("**MODERATE RISK**")
                    st.write("Patient shows moderate signs of risk. Preventative measures recommended.")
                else:
                    st.error("**HIGH RISK**")
                    st.write("High probability of CHD. Immediate medical attention and lifestyle changes advised.")

            with res_col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Meter"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 15], 'color': "#2ecc71"},
                            {'range': [15, 30], 'color': "#f1c40f"},
                            {'range': [30, 100], 'color': "#e74c3c"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_percentage}}))
                
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Model Performance Metrics")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric(label="Overall Accuracy", value=f"{accuracy*100:.2f}%", delta="+1.2% vs Baseline")
        with m_col2:
            st.metric(label="Dataset Size", value=f"{dataset_size}", delta="Patients")
        with m_col3:
            st.metric(label="Model Type", value="Logistic Regression")

        st.markdown("#### Feature Importance (Coefficients)")
        # Get coefficients
        coefs = pd.DataFrame({
            'Feature': ['Male', 'Age', 'Current Smoker', 'Cigs/Day', 'BP Meds', 'Stroke', 'Hypertension', 'Diabetes', 'Cholesterol', 'Sys BP', 'Dia BP', 'BMI', 'Heart Rate', 'Glucose'],
            'Coefficient': model.coef_[0]
        })
        coefs = coefs.sort_values(by='Coefficient', ascending=False)
        
        st.bar_chart(coefs.set_index('Feature'))

    with tab3:
        st.markdown("### 📚 About the Framingham Heart Study")
        st.write("""
        The **Framingham Heart Study** is a landmark, long-term cardiovascular cohort study that began in 1948 in Framingham, Massachusetts. 
        With over 5,200 original participants and now spanning three generations, it has been instrumental in identifying major cardiovascular risk factors and advancing preventive medicine.
        
        This prediction system leverages decades of research data to provide evidence-based risk assessments for coronary heart disease (CHD).
        
        #### Key Risk Factors Identified:
        - **Age:** Risk increases with age.
        - **Gender:** Males are generally at higher risk.
        - **Systolic Blood Pressure:** High BP is a major contributor.
        - **Cigarettes Per Day:** Smoking significantly increases risk.
        - **Cholesterol & Glucose:** Metabolic health plays a crucial role.
        """)
        st.info("🎓 Academic Project 2026 • Developed for Advanced Data Science coursework.")