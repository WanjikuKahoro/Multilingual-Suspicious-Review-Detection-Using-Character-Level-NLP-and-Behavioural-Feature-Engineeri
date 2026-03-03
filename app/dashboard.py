import streamlit as st
import joblib
import pandas as pd
import os
import sys
import numpy as np

# 1. FIX NUMPY COMPATIBILITY
# Re-adding attributes removed in newer NumPy versions to prevent crashes in older libraries
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

# 2. SET UP SYSTEM PATHS
# This ensures Python can find 'feature_engineering.py' inside the 'app' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from feature_engineering import build_features

# 3. CONFIGURE PAGE
st.set_page_config(page_title="Safaricom Review Audit", layout="centered", page_icon="🛡️")

# 4. LOAD MODEL WITH ABSOLUTE PATH
@st.cache_resource
def load_assets():
    # Find the 'model' folder by going up one level from 'app'
    repo_root = os.path.dirname(current_dir)
    model_path = os.path.join(repo_root, "model", "model.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    model = joblib.load(model_path)
    threshold = 0.95  # Confidence threshold for flagging reviews
    return model, threshold

# Execute loading
try:
    model, threshold = load_assets()
except Exception as e:
    st.error(f"❌ Initialization Error: {e}")
    st.stop()

# 5. USER INTERFACE
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

# 6. PREDICTION LOGIC
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
            probability = model.predict_proba(features)[:, 1][0]
        
        # Display Results
        st.divider()
        if probability >= threshold:
            st.error(f"🚩 **SUSPICIOUS**")
            st.metric("Anomaly Probability", f"{probability:.2%}")
            st.info(f"This review exceeds the audit threshold of {threshold:.0%}.")
        else:
            st.success(f"✅ **GENUINE**")
            st.metric("Anomaly Probability", f"{probability:.2%}")