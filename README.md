# Model Diagnostics Studio

Streamlit app for dataset upload, target/predictor selection, model fitting, validation metrics, and residual diagnostics.

## Features
- CSV and Excel upload
- Choose dependent variable and multiple independent variables
- Classification and regression models
- Validation metrics
- Residual distribution and residuals-versus-predictions plots
- For binary classification: ROC, precision–recall, calibration, and confusion matrix
- Download validation predictions and residuals as CSV
- Configurable row cap for resource control

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
Push `app.py`, `requirements.txt`, and `README.md` to GitHub, then create a Streamlit app using `app.py` as the entry point. Upload data through the app rather than committing private data or credentials to the repository.

## Notes
Binary classification is supported. For credit-risk evaluation, prefer a time-based validation split for final analysis. This app is for exploratory diagnostics and not production lending decisions.
