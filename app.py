import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
st.set_page_config(page_title="Model Diagnostics Studio", page_icon="📊", layout="wide")
st.title("Model Diagnostics Studio")
st.caption("Upload a dataset, select the target and predictors, fit a model, and inspect validation metrics and residual diagnostics.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.divider()
    st.header("Model setup")
    task = st.radio("Problem type", ["Classification", "Regression"])
    test_size = st.slider("Validation size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    max_rows = st.number_input("Maximum rows to use (0 = all)", min_value=0, max_value=5_000_000, value=100_000, step=10_000)

@st.cache_data(show_spinner=False)
def read_data(file_bytes, filename):
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(buf, low_memory=False)
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, encoding="latin-1", low_memory=False)
    return pd.read_excel(buf)

def make_pipeline(task_name, model_name, X_train):
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(transformers)
    if task_name == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=42),
            "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, random_state=42),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=3, n_jobs=-1, random_state=42),
            "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, random_state=42),
        }
    return Pipeline([("preprocess", prep), ("model", models[model_name])])

if uploaded is None:
    st.info("Upload a CSV or Excel file to begin. No dataset needs to be stored in the public GitHub repository.")
    with st.expander("Example data format"):
        st.code("age,income,previous_defaults,target\n34,55000,0,0\n51,42000,1,1\n29,68000,0,0", language="csv")
        st.caption("For credit risk, use a binary target such as target (0 = non-default, 1 = default).")
    st.stop()

try:
    df = read_data(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()
if df.columns.duplicated().any():
    st.error("Column names must be unique. Rename duplicate columns and upload again.")
    st.stop()

st.subheader("Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", f"{df.shape[1]:,}")
c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
st.dataframe(df.head(12), use_container_width=True, hide_index=True)
if len(df) > 250_000:
    st.warning("Large dataset detected. Set a row limit or upload a representative sample to avoid memory/time limits.")

st.subheader("Choose target and predictors")
columns = list(df.columns)
target_default = columns.index("target") if "target" in columns else len(columns) - 1
target = st.selectbox("Dependent variable (target)", columns, index=target_default)
predictor_options = [col for col in columns if col != target]
features = st.multiselect("Independent variables (predictors)", predictor_options, default=predictor_options[:min(10, len(predictor_options))])
if task == "Classification":
    model_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Histogram Gradient Boosting"])
else:
    model_name = st.selectbox("Model", ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Histogram Gradient Boosting"])
stratify = st.checkbox("Stratify split (classification)", value=True, disabled=(task != "Classification"))
fit = st.button("Fit model and show diagnostics", type="primary", use_container_width=True)

if fit:
    if not features:
        st.error("Select at least one predictor.")
        st.stop()
    data = df[[target] + features].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    usable = [col for col in features if data[col].notna().sum() > 0 and data[col].nunique(dropna=True) > 1]
    dropped = sorted(set(features) - set(usable))
    if dropped:
        st.warning("Dropped constant or entirely missing predictors: " + ", ".join(dropped))
    if not usable:
        st.error("No usable predictors remain.")
        st.stop()
    X = data[usable].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique(dropna=True) > 30:
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().mean() >= 0.9:
                X[col] = converted
    raw_y = data[target]
    if task == "Classification":
        if raw_y.nunique() != 2:
            st.error(f"Binary classification only: target '{target}' has {raw_y.nunique()} classes.")
            st.stop()
        vals = list(raw_y.dropna().unique())
        if set(vals) == {0, 1} or set(vals) == {False, True}:
            y = raw_y.astype(int)
        else:
            ordered = sorted(vals, key=str)
            y = raw_y.map({ordered[0]: 0, ordered[1]: 1}).astype(int)
            st.caption(f"Target encoding: {ordered[0]!r} → 0, {ordered[1]!r} → 1; positive class is {ordered[1]!r}.")
        if y.value_counts().min() < 2:
            st.error("Each class needs at least two observations.")
            st.stop()
    else:
        y = pd.to_numeric(raw_y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if y.nunique() < 2:
            st.error("Regression target must be numeric and have at least two distinct values.")
            st.stop()
    if max_rows and len(X) > max_rows:
        rng = np.random.RandomState(int(random_state))
        if task == "Classification":
            sample_indices = []
            for _, group in y.groupby(y).groups.items():
                group_idx = np.asarray(list(group))
                n = max(1, round(int(max_rows) * len(group_idx) / len(X)))
                sample_indices.extend(rng.choice(group_idx, size=min(n, len(group_idx)), replace=False).tolist())
            if len(sample_indices) > int(max_rows):
                sample_indices = rng.choice(sample_indices, size=int(max_rows), replace=False).tolist()
            idx = X.loc[sample_indices].index
        else:
            idx = X.sample(n=int(max_rows), random_state=int(random_state)).index
        X, y = X.loc[idx], y.loc[idx]
        st.info(f"Using {len(X):,} rows for this run.")
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size / 100, random_state=int(random_state),
            stratify=(y if task == "Classification" and stratify and y.value_counts().min() >= 2 else None)
        )
        model = make_pipeline(task, model_name, X_train)
        with st.spinner("Training model and calculating validation diagnostics..."):
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            if task == "Classification":
                probs = model.predict_proba(X_val)[:, 1]
                residual = np.asarray(y_val, dtype=float) - probs
            else:
                probs = None
                residual = np.asarray(y_val, dtype=float) - np.asarray(pred, dtype=float)
        st.success(f"Model trained · {len(X_train):,} training rows · {len(X_val):,} validation rows")
        st.subheader("Validation metrics")
        if task == "Classification":
            auc = roc_auc_score(y_val, probs)
            ap = average_precision_score(y_val, probs)
            br = brier_score_loss(y_val, probs)
            metrics = st.columns(6)
            for col, label, value in zip(metrics, ["ROC-AUC", "Gini", "Average precision", "Brier score", "Accuracy", "F1 score"],
                [f"{auc:.4f}", f"{2*auc-1:.4f}", f"{ap:.4f}", f"{br:.5f}", f"{accuracy_score(y_val,pred):.4f}", f"{f1_score(y_val,pred,zero_division=0):.4f}"]):
                col.metric(label, value)
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{precision_score(y_val,pred,zero_division=0):.4f}")
            m2.metric("Recall", f"{recall_score(y_val,pred,zero_division=0):.4f}")
            m3.metric("Positive class prevalence", f"{np.mean(y_val):.2%}")
            st.caption("Metrics use the held-out validation set. Average precision depends on class prevalence; lower Brier score is better.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_val,pred)):.4f}")
            m2.metric("MAE", f"{mean_absolute_error(y_val,pred):.4f}")
            m3.metric("R²", f"{r2_score(y_val,pred):.4f}")

        st.subheader("Residual diagnostics")
        st.caption("Residual = observed − predicted. For binary classification, predicted means the estimated probability of class 1.")
        diag = pd.DataFrame({"Observed": np.asarray(y_val), "Predicted": probs if task == "Classification" else np.asarray(pred), "Residual": residual})
        left, right = st.columns(2)
        with left:
            fig = px.histogram(diag, x="Residual", nbins=50, marginal="box", title="Residual distribution")
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(xaxis_title="Observed − predicted", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(diag, x="Predicted", y="Residual", opacity=0.45, title="Residuals vs predicted")
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(xaxis_title="Predicted probability" if task == "Classification" else "Predicted value")
            st.plotly_chart(fig, use_container_width=True)

        if task == "Classification":
            left, right = st.columns(2)
            with left:
                fpr, tpr, _ = roc_curve(y_val, probs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC = {auc:.3f}"))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line=dict(dash="dash")))
                fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                precision, recall, _ = precision_recall_curve(y_val, probs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"AP = {ap:.3f}"))
                fig.add_hline(y=float(np.mean(y_val)), line_dash="dash", annotation_text="Positive prevalence")
                fig.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig, use_container_width=True)
            display = CalibrationDisplay.from_predictions(y_val, probs, n_bins=10, strategy="quantile")
            display.figure_.suptitle("Calibration curve")
            st.pyplot(display.figure_, clear_figure=True)
            st.markdown("**Confusion matrix (threshold = 0.50)**")
            cm = confusion_matrix(y_val, (probs >= 0.5).astype(int), labels=[0, 1])
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), use_container_width=True)
        with st.expander("Run details"):
            st.write({"model": model_name, "problem_type": task, "target": target, "predictors": usable, "training_rows": len(X_train), "validation_rows": len(X_val), "seed": int(random_state)})
        out = diag.copy()
        out.insert(0, "row_index", X_val.index.astype(str))
        st.download_button("Download validation predictions and residuals (CSV)", out.to_csv(index=False).encode("utf-8"), "validation_diagnostics.csv", "text/csv", use_container_width=True)
    except Exception as exc:
        st.error(f"Model run failed: {exc}")
        st.info("Try fewer rows, fewer predictors, or remove ID/high-cardinality columns. Ensure the selected target matches the problem type.")

with st.expander("Important modelling notes"):
    st.markdown("""
- Exclude unique identifiers, target leakage, and columns unavailable at prediction time.
- For credit-risk work, a time-based validation split is often more appropriate than a random split.
- Classification residuals here are observed 0/1 values minus predicted probabilities; they are not ordinary continuous-regression residuals.
- The confusion matrix uses a 0.50 probability threshold. Select a threshold using business costs and validation evidence.
- This is an exploratory tool, not a production credit decision system. Do not upload sensitive personal data to a public deployment.
""")
