import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Credit Risk Analytics Dashboard")
st.caption("Experimental credit-risk model — not a lending decision system")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_credit_risk_model.joblib")
    calibrator = joblib.load("models/pd_sigmoid_calibrator.joblib")
    with open("models/credit_risk_features.json") as f:
        features = json.load(f)
    return model, calibrator, features

model, calibrator, features = load_artifacts()

st.subheader("Applicant feature input")
st.write(
    "Enter the engineered features expected by the model. "
    "Values must use the same definitions and units as the training pipeline."
)

with st.form("scoring_form"):
    values = {}
    for feature in features:
        values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.4f"
        )
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    X = pd.DataFrame([[values[f] for f in features]], columns=features)
    raw_score = model.predict_proba(X)[:, 1]
    calibrated_pd = calibrator.predict_proba(
        raw_score.reshape(-1, 1)
    )[:, 1][0]

    st.metric("Calibrated probability of default", f"{calibrated_pd:.2%}")
    st.metric("Raw model score", f"{raw_score[0]:.4f}")

    st.warning(
        "This is an experimental estimate. Zero-filled inputs may not "
        "represent a real applicant. Do not use it for actual lending decisions."
    )
