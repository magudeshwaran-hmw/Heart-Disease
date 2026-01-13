import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Title Sections */
        .title-container {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            padding-top: 2rem;
        }
        .title-text {
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .subtitle-text {
            font-size: 1.2rem;
            font-weight: 300;
            margin-top: 0.5rem;
            opacity: 0.9;
        }

        /* Badge */
        .badge-container {
            display: flex;
            justify-content: center;
            margin-bottom: 3rem;
        }
        .badge {
            background: #FFB7B2;
            color: #d32f2f;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Form Container */
        .form-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            margin-bottom: 2rem;
        }

        /* Customizing Input Labels to be white */
        .stNumberInput label, .stSelectbox label {
            color: white !important;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(90deg, #ff8a00 0%, #e52e71 100%);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(229, 46, 113, 0.4);
            color: white;
        }

        /* About Section Card */
        .about-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            color: #333;
            margin-top: 3rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .about-title {
            color: #764ba2;
            font-weight: 700;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        .disclaimer {
            font-size: 0.8rem;
            color: #666;
            margin-top: 1rem;
            font-style: italic;
        }
        
        /* Icon styling in header */
        .header-icon {
            font-size: 4rem;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_artifacts():
    try:
        with open("heart_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

# --- HEADER SECTION ---
st.markdown("""
<div class="title-container">
    <div class="header-icon">🩺</div>
    <div class="title-text">AI Heart Disease Prediction</div>
    <div class="subtitle-text">Advanced Machine Learning for Health & Bioinformatics</div>
</div>
""", unsafe_allow_html=True)

# --- BADGE SECTION ---
st.markdown("""
<div class="badge-container">
    <div class="badge">⚛️ Powered by Advanced Logistic Regression & Random Forest Models</div>
</div>
""", unsafe_allow_html=True)

# --- MAIN FORM SECTION ---
if model and scaler:
    with st.container():
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Patient Health Information")
        st.markdown("<p style='opacity: 0.8; font-size: 0.9rem; margin-bottom: 20px;'>Enter the patient's medical parameters below for AI-powered heart disease risk assessment</p>", unsafe_allow_html=True)
        
        with st.form("heart_form"):
            col1, col2 = st.columns(2, gap="large")

            with col1:
                age = st.number_input("👤 Age (years)", min_value=1, max_value=120, value=50)
                sex_val = st.selectbox("⚧ Sex", options=["Male", "Female"])
                cp_val = st.selectbox("💔 Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
                trestbps = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
                chol = st.number_input("🥓 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
                fbs_val = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", options=["True", "False"])
                restecg_val = st.selectbox("💓 Resting ECG Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])

            with col2:
                thalach = st.number_input("📈 Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
                exang_val = st.selectbox("🏃 Exercise Induced Angina", options=["Yes", "No"])
                oldpeak = st.number_input("📉 ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
                slope_val = st.selectbox("📐 Slope of Peak Exercise ST", options=["Upsloping", "Flat", "Downsloping"])
                ca = st.selectbox("🩺 Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
                thal_val = st.selectbox("🧬 Thalassemia", options=["Normal", "Fixed Defect", "Reversable Defect"])

            # Map inputs to model values
            sex = 1 if sex_val == "Male" else 0
            cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp_val)
            fbs = 1 if fbs_val == "True" else 0
            restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg_val)
            exang = 1 if exang_val == "Yes" else 0
            slope = ["Upsloping", "Flat", "Downsloping"].index(slope_val)
            thal = ["Normal", "Fixed Defect", "Reversable Defect"].index(thal_val) + 1  # 1-indexed in many datasets

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("ANALYZE HEART DISEASE RISK")
        
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            # Prepare input data
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Scale the input
            scaled_data = scaler.transform(input_data)

            # Predict
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)[0][1] * 100

            # Result Display
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.markdown(f"""
                <div style="background-color: #ffcccc; color: #cc0000; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #ff0000; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="margin:0; font-size: 1.5rem;">🚨 High Risk of Heart Disease</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">Probability: <strong>{probability:.2f}%</strong></p>
                    <p>Please consult a cardiologist immediately for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="margin:0; font-size: 1.5rem;">✅ Low Risk of Heart Disease</h2>
                    <p style="font-size: 1.2rem; margin-top: 10px;">Probability: <strong>{probability:.2f}%</strong></p>
                    <p>Keep maintaining a healthy lifestyle!</p>
                </div>
                """, unsafe_allow_html=True)

    # --- ABOUT SECTION ---
    st.markdown("""
    <div class="about-card">
        <div class="about-title">📊 About This System</div>
        <p>This AI-powered system uses advanced machine learning models (Logistic Regression, Random Forest, XGBoost) trained on clinical patient records. The model analyzes key health parameters to predict heart disease risk with high accuracy. This tool is designed to assist healthcare professionals in early detection and risk assessment.</p>
        <div class="disclaimer">⚠️ Disclaimer: This is a predictive tool and should not replace professional medical diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Model files not found! Please run `python model.py` first to train and save the models.")
