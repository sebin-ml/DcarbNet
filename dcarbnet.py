# streamlit_app.py
import streamlit as st
import joblib
import numpy as np

# Title
st.title("🔍 2D Material Screening for H₂ Storage and CO₂ Capture")

# Sidebar for input
st.sidebar.header("🧪 Enter Material Properties")
surface_area = st.sidebar.number_input("Surface Area (m²/g)", min_value=0.0, max_value=5000.0, value=1000.0)
bandgap = st.sidebar.number_input("Bandgap (eV)", min_value=0.0, max_value=5.0, value=1.2)

# Load models
model_h2 = joblib.load("model_h2.joblib")
model_co2 = joblib.load("model_co2.joblib")

# Main section
features = np.array([[surface_area, bandgap]])

if st.button("🎯 Predict H₂ and CO₂ Properties"):
    pred_h2 = model_h2.predict(features)[0]
    pred_co2 = model_co2.predict(features)[0]
    
    st.markdown("### 🔍 Predicted Results")
    st.metric(label="H₂ Adsorption Energy (eV)", value=f"{pred_h2:.3f}")
    st.metric(label="CO₂ Binding Energy (eV)", value=f"{pred_co2:.3f}")

    # Evaluation ranges
    st.markdown("### 🧮 Evaluation Against Ideal Ranges")

    # H₂ evaluation
    if -0.6 <= pred_h2 <= -0.2:
        st.success("✅ H₂ adsorption energy is within the ideal range (-0.6 to -0.2 eV).")
    else:
        st.warning("⚠️ H₂ adsorption energy is outside the ideal range.")

    # CO₂ evaluation
    if -1.0 <= pred_co2 <= -0.3:
        st.success("✅ CO₂ binding energy is within the ideal range (-1.0 to -0.3 eV).")
    else:
        st.warning("⚠️ CO₂ binding energy is outside the ideal range.")

    # Notes
    st.markdown("---")
    st.markdown("📘 **Ideal Adsorption Guidelines**")
    st.markdown("""
    - H₂ adsorption energy should be moderate (−0.2 to −0.6 eV) for reversible storage.
    - CO₂ binding energy depends on the process (physisorption vs chemisorption) but is typically ideal in the −0.3 to −1.0 eV range for selectivity and recyclability.
    """)

