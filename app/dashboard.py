import streamlit as st
import joblib
import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.utils.validation import check_is_fitted

import sklearn, joblib, numpy, scipy
st.write("numpy:", numpy.__version__)
st.write("scipy:", scipy.__version__)
st.write("sklearn:", sklearn.__version__)
st.write("joblib:", joblib.__version__)
st.write("✅ dashboard.py started")

# NumPy compatibility (optional)
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from feature_engineering import build_features

st.set_page_config(page_title="Safaricom Review Audit", layout="centered", page_icon="🛡️")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent   # go from app/ → project root

@st.cache_resource
def load_assets():
    model_path = PROJECT_ROOT / "model" / "model_clean.joblib"
    thresh_path = PROJECT_ROOT / "model" / "threshold.joblib"

    model = joblib.load(model_path)
    threshold = float(joblib.load(thresh_path))

    # normalize threshold (handles numpy scalar/array)
    try:
        threshold = float(threshold)
    except TypeError:
        threshold = float(threshold[0])

    # sanity: ensure fitted
    check_is_fitted(model.named_steps["clf"])

    return model, threshold

try:
    model, threshold = load_assets()

    # DEBUG (temporary)
    st.write("Model type:", type(model))
    st.write("Model object:", model)

except Exception as e:
    st.error(f"❌ Initialization Error: {e}")
    st.stop()

#  USER INTERFACE
st.title("🛡️ Safaricom Review Shield")
st.markdown("Analyzing English, Swahili, and Sheng reviews for machine-learning based anomaly detection.")

with st.form("input_form"):
    review_text = st.text_area("Review Text", placeholder="Enter review here (e.g., 'This app is great' or 'Pesa imepotea...')")
    
    col1, col2 = st.columns(2)
    with col1:
        rating = st.slider("User Rating", 1, 5, 3)
        thumbs = st.number_input("Helpful Votes (Thumbs Up)", min_value=0, step=1)
    with col2:
        is_mixed = st.toggle("Code Mixed (Eng/Swa)")
        is_sheng = st.toggle("Sheng-like Language")
    
    submit = st.form_submit_button("Analyze Review", use_container_width=True)

#  PREDICTION LOGIC
if submit:
    if not review_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Prepare input data for the model
        input_data = pd.DataFrame([{
            "review_text": review_text,
            "rating": rating,
            "thumbs_up": thumbs,
            "is_code_mixed": int(is_mixed),
            "is_sheng_like": int(is_sheng)
        }])
        
        # Transform and Predict
        with st.spinner("Analyzing patterns..."):
            features = build_features(input_data)
            probability = float(model.predict_proba(features)[:, 1][0])
        
        # Display Results
        st.divider()
        if probability >= threshold:
            st.error(f"🚩 **SUSPICIOUS**")
            st.metric("Anomaly Probability", f"{probability:.2%}")
            st.info(f"This review exceeds the audit threshold of {threshold:.0%}.")
        else:
            st.success(f"✅ **GENUINE**")
            st.metric("Anomaly Probability", f"{probability:.2%}")
