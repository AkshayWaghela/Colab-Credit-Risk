# Credit Risk Scoring & Probability of Default

An experimental credit-risk analytics project using engineered previous-application
features, XGBoost, and sigmoid (Platt) calibration.

## Validation snapshot

- Time-based holdout: July–October 2020
- ROC-AUC: approximately 0.706
- Gini: approximately 0.412
- Average precision: approximately 0.053
- Brier score: approximately 0.0204

These are historical holdout results, not a guarantee of future performance.
The low default prevalence also means average precision should be interpreted
against the baseline prevalence.

## Project components

- `app.py` — Streamlit scoring demonstration
- `models/` — trained XGBoost model, sigmoid calibrator, and ordered feature list
- `requirements.txt` — deployment dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important limitations

This is a portfolio/educational demonstration, not a production credit decision
system. The app's manually entered values are not a substitute for the original
feature-engineering pipeline. Do not enter real customer information or use
outputs to make lending decisions. Model validation, calibration monitoring,
fairness assessment, governance, and independent review would be required before
any real-world use.

## Data and reproducibility

Datasets and generated Parquet files are intentionally excluded from this
repository. Recreate the engineered features using the original project pipeline
and the relevant dataset documentation before interpreting model outputs.

## Author

Akshay Waghela
