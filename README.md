# Credit Risk Scoring & Probability of Default

An experimental credit-risk analytics project using engineered previous-application
features, XGBoost, and sigmoid probability calibration.

## Model validation snapshot

- Time-based validation period: July–October 2020
- ROC-AUC: approximately 0.706
- Average precision: approximately 0.053
- Calibrated Brier score: approximately 0.0204

These results are dataset-specific and do not establish production suitability.
The low default prevalence means precision-recall performance is important.
Calibration and model performance may change across populations and time.

## Technology

Python, Pandas, scikit-learn, XGBoost, joblib, Streamlit.

## Repository contents

- `app.py` — Streamlit demonstration
- `models/` — trained model, matching sigmoid calibrator, feature order
- `requirements.txt` — deployment dependencies

The raw and processed datasets are intentionally excluded from this repository.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy this repository using Streamlit Community Cloud and select `app.py`
as the app entry point.

## Important limitations

This is an educational portfolio project, not a lending decision system.
The demo's manually entered engineered features may not correspond to valid
applicant records. Do not use its estimates for real credit decisions.

The XGBoost model and sigmoid calibrator must be used together. The calibrator
was fitted for this specific model and training procedure.

## Author

Akshay Waghela
