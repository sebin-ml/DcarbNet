# streamlit_app.py
import streamlit as st
import joblib
import numpy as np

st.title("DcarbNet: 2D Material Screening for H₂ Storage and CO₂ Capture")

st.markdown("### Enter Material Features")
st.markdown("####### Dr. Sebin Devasia")
surface_area = st.number_input("Surface Area (m²/g)", min_value=0.0, max_value=5000.0, value=1000.0)
bandgap = st.number_input("Bandgap (eV)", min_value=0.0, max_value=5.0, value=1.2)

features = np.array([[surface_area, bandgap]])

# Load models
model_h2 = joblib.load("model_h2.joblib")
model_co2 = joblib.load("model_co2.joblib")

if st.button("Predict Properties"):
    pred_h2 = model_h2.predict(features)[0]
    pred_co2 = model_co2.predict(features)[0]
    
    st.success(f"H₂ adsorption energy predicted: {pred_h2:.3f} eV")
    st.success(f"CO₂ binding energy predicted: {pred_co2:.3f} eV")

    st.markdown("---")
    st.markdown("**Ideal Ranges**")
    st.markdown("- H₂ adsorption: -0.2 to -0.6 eV")
    st.markdown("- CO₂ binding: -0.3 to -1.0 eV (material-dependent)")

