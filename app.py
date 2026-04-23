# ============================================================
#  Heart Disease Prediction - Streamlit App
#  Uses joblib to load the saved model
# ============================================================

import streamlit as st
import joblib
import numpy as np

# ── 1. PAGE SETUP ───────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ── 2. LOAD MODEL WITH JOBLIB ───────────────────────────────
# joblib.load()  →  reads the saved .pkl file back into memory
@st.cache_resource          # cache so it loads only ONCE
def load_model():
    model = joblib.load("heart_model.pkl")   # <── joblib loads the model
    return model

model = load_model()

# ── 3. TITLE & DESCRIPTION ──────────────────────────────────
st.title("❤️ Heart Disease Prediction")
st.markdown("Fill in the patient details below and click **Predict** to check the risk.")
st.divider()

# ── 4. INPUT FIELDS ─────────────────────────────────────────
st.subheader("🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age       = st.number_input("Age",               min_value=1,  max_value=120, value=50)
    sex       = st.selectbox("Sex",                  options=[0, 1],
                             format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    cp        = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3],
                             help="0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic")
    trestbps  = st.number_input("Resting Blood Pressure",  min_value=80,  max_value=250, value=120)
    chol      = st.number_input("Cholesterol (mg/dl)",     min_value=100, max_value=600, value=200)
    fbs       = st.selectbox("Fasting Blood Sugar > 120",  options=[0, 1],
                             format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    restecg   = st.selectbox("Resting ECG Results",        options=[0, 1, 2])

with col2:
    thalach   = st.number_input("Max Heart Rate Achieved", min_value=60,  max_value=250, value=150)
    exang     = st.selectbox("Exercise Induced Angina",    options=[0, 1],
                             format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    oldpeak   = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0,
                                value=1.0, step=0.1)
    slope     = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                             help="0=Upsloping, 1=Flat, 2=Downsloping")
    ca        = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal      = st.selectbox("Thal",                        options=[0, 1, 2, 3],
                             help="0=Normal, 1=Fixed Defect, 2=Reversible Defect")

st.divider()

# ── 5. PREDICT BUTTON ───────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):

    # Put all inputs into a numpy array (same order as training columns)
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # joblib-loaded model → call .predict() and .predict_proba()
    prediction   = model.predict(input_data)[0]          # 0 or 1
    probability  = model.predict_proba(input_data)[0]    # [prob_0, prob_1]

    risk_percent = round(probability[1] * 100, 1)

    st.subheader("📊 Result")

    if prediction == 1:
        st.error(f"⚠️  **High Risk of Heart Disease**  —  Confidence: {risk_percent}%")
    else:
        st.success(f"✅  **Low Risk of Heart Disease**  —  Confidence: {round(probability[0]*100,1)}%")

    # Show probability bar
    st.metric("Heart Disease Probability", f"{risk_percent}%")
    st.progress(int(risk_percent))

    st.caption("_This is a screening tool only — always consult a medical professional._")

