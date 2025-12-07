import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from io import StringIO
from typing import Optional, Dict

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    PowerTransformer,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.inspection import partial_dependence

import shap


# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="CMPE 255 – Auto Data Toolkit (Enhanced)",
    layout="wide",
    page_icon="🚢",
)

st.title("🚢 CMPE 255 – Auto Data Toolkit (Enhanced)")
st.markdown(
    """
This app implements an **end-to-end CRISP-DM** workflow with your requested features:

1. Multiple **imputation** strategies (mean / median / KNN / iterative)  
2. **Outlier** removal (IQR / IsolationForest)  
3. **Skew correction** (log1p / Yeo-Johnson)  
4. **Categorical encoding** (One-Hot / Ordinal)  
5. **Datetime feature engineering** (year / month / day)  
6. **Duplicate removal**  
7. **Feature selection** (VarianceThreshold / RFE)  
8. **Explainability** (feature importance, SHAP top-10, PDP)  
9. Downloadable **cleaned dataset** and **HTML report**  
"""
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
@st.cache_data
def load_titanic_demo() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df


def expand_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Detect datetime-like columns and expand them into year/month/day."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        dt = None

        if np.issubdtype(s.dtype, np.datetime64):
            dt = s
        elif s.dtype == "object":
            try:
                dt_candidate = pd.to_datetime(
                    s, errors="raise", infer_datetime_format=True)
                if np.issubdtype(dt_candidate.dtype, np.datetime64):
                    dt = dt_candidate
            except Exception:
                dt = None

        if dt is not None:
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df = df.drop(columns=[col])

    return df


def remove_outliers(df: pd.DataFrame, numeric_cols, method: str):
    """Apply IQR or IsolationForest-based outlier removal."""
    if method == "None" or not numeric_cols:
        return df, 0

    df = df.copy()
    initial_rows = len(df)
    X_num = df[numeric_cols]

    if method == "IQR":
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            col_data = X_num[col]
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask &= col_data.between(lower, upper) | col_data.isna()
        df = df[mask]

    elif method == "IsolationForest":
        non_na = X_num.dropna()
        if len(non_na) > 0:
            iso = IsolationForest(random_state=42, contamination="auto")
            iso.fit(non_na)
            preds = iso.predict(non_na)  # 1 inlier, -1 outlier
            keep_idx = non_na.index[preds == 1]
            df = df.loc[keep_idx]

    removed = initial_rows - len(df)
    return df, removed


def build_preprocessor(
    X: pd.DataFrame,
    imputation_strategy: str,
    skew_strategy: str,
    cat_encoding: str,
    variance_threshold: Optional[float],
    rfe_n_features: Optional[int],
):
    """Build ColumnTransformer + feature selection + RandomForest pipeline."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # ---- Imputers ----
    if imputation_strategy == "Mean":
        num_imputer = SimpleImputer(strategy="mean")
    elif imputation_strategy == "Median":
        num_imputer = SimpleImputer(strategy="median")
    elif imputation_strategy == "KNN":
        num_imputer = KNNImputer(n_neighbors=5)
    else:  # Iterative
        num_imputer = IterativeImputer(random_state=42)

    cat_imputer = SimpleImputer(strategy="most_frequent")

    # ---- Skew correction ----
    skew_step = None
    if skew_strategy == "Log1p":
        skew_step = ("skew", FunctionTransformer(
            lambda x: np.log1p(np.clip(x, a_min=0, a_max=None))))
    elif skew_strategy == "Yeo-Johnson":
        skew_step = ("skew", PowerTransformer(method="yeo-johnson"))

    # ---- Categorical encoding ----
    if cat_encoding == "One-Hot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)

    num_steps = [("imputer", num_imputer)]
    if skew_step is not None:
        num_steps.append(skew_step)
    num_steps.append(("scaler", StandardScaler()))
    num_tf = Pipeline(num_steps)

    cat_tf = Pipeline(
        [
            ("imputer", cat_imputer),
            ("encoder", encoder),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    steps = [("pre", pre)]

    if variance_threshold is not None and variance_threshold > 0.0:
        steps.append(("var_sel", VarianceThreshold(
            threshold=variance_threshold)))

    if rfe_n_features is not None and rfe_n_features > 0:
        base_estimator = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
        )
        rfe = RFE(base_estimator, n_features_to_select=rfe_n_features, step=1)
        steps.append(("rfe", rfe))

    steps.append(("model", model))
    pipe = Pipeline(steps)

    return pipe, numeric_cols, categorical_cols


def generate_html_report(
    config: Dict[str, object],
    metrics: Dict[str, float],
    classification_rep: str,
) -> str:
    """Create a lightweight HTML report summarizing config + metrics."""
    html = StringIO()
    html.write("<html><head><title>Auto Data Toolkit Report</title></head><body>")
    html.write("<h1>Auto Data Toolkit – Model Report</h1>")
    html.write("<h2>Configuration</h2><ul>")
    for k, v in config.items():
        html.write(f"<li><b>{k}</b>: {v}</li>")
    html.write("</ul>")
    html.write("<h2>Metrics</h2><ul>")
    for k, v in metrics.items():
        html.write(f"<li><b>{k}</b>: {v:.4f}</li>")
    html.write("</ul>")
    html.write("<h2>Classification Report</h2><pre>")
    html.write(classification_rep)
    html.write("</pre>")
    html.write("<p><i>Figures such as confusion matrix, ROC curve, SHAP plots, "
               "and PDPs are available in the app and in the MLflow pipeline.</i></p>")
    html.write("</body></html>")
    return html.getvalue()


# ---------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio(
    "Choose dataset:",
    ["Titanic demo", "Upload CSV"],
    index=0,
)

if data_source == "Titanic demo":
    raw_df = load_titanic_demo()
    st.sidebar.info("Using Titanic demo dataset from GitHub.")
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("👈 Upload a CSV file in the sidebar to begin.")
        st.stop()
    raw_df = pd.read_csv(uploaded)
    st.sidebar.success("Custom CSV loaded.")

st.subheader("📊 Raw Data Preview")
st.write(f"Shape: **{raw_df.shape[0]} rows × {raw_df.shape[1]} columns**")
st.dataframe(raw_df.head(), use_container_width=True)

# ---- Cleaning options ----
st.sidebar.header("2. Data Cleaning & Preparation")

remove_dupes = st.sidebar.checkbox("Remove duplicate rows", value=True)

imputation_strategy = st.sidebar.selectbox(
    "Numeric imputation strategy",
    ["Mean", "Median", "KNN", "Iterative"],
    index=0,
)

outlier_method = st.sidebar.selectbox(
    "Outlier removal",
    ["None", "IQR", "IsolationForest"],
    index=0,
)

skew_strategy = st.sidebar.selectbox(
    "Skew correction (numeric)",
    ["None", "Log1p", "Yeo-Johnson"],
    index=0,
)

cat_encoding = st.sidebar.selectbox(
    "Categorical encoding",
    ["One-Hot", "Ordinal"],
    index=0,
)

variance_threshold_val = st.sidebar.slider(
    "VarianceThreshold (0 disables)",
    min_value=0.0,
    max_value=0.2,
    value=0.0,
    step=0.01,
)

rfe_n_features_val = st.sidebar.slider(
    "RFE: number of features (0 disables)",
    min_value=0,
    max_value=50,
    value=0,
    step=1,
)

st.sidebar.header("3. Target & Split")

default_target_candidates = [
    c for c in raw_df.columns if c.lower() in ["survived", "target", "label", "class", "outcome", "y"]
]
if default_target_candidates:
    default_target = default_target_candidates[0]
else:
    default_target = raw_df.columns[-1]

target_col = st.sidebar.selectbox(
    "Target column",
    raw_df.columns.tolist(),
    index=raw_df.columns.get_loc(default_target),
)

test_size = st.sidebar.slider("Test size fraction", 0.1, 0.4, 0.2, step=0.05)

run_button = st.sidebar.button("🚀 Run Auto-Toolkit")

if not run_button:
    st.info("👈 Configure options and click **Run Auto-Toolkit**.")
    st.stop()

# ---------------------------------------------------------------
# Cleaning and preprocessing at dataframe level
# ---------------------------------------------------------------
df = raw_df.copy()

if remove_dupes:
    before = len(df)
    df = df.drop_duplicates()
    removed_dup = before - len(df)
    st.write(f"✅ Removed {removed_dup} duplicate rows.")

# Drop rows with missing target
df = df.dropna(subset=[target_col]).copy()

# Datetime features
df = expand_datetime_features(df)

# Outliers
numeric_cols_all = df.drop(columns=[target_col]).select_dtypes(
    include=["number"]).columns.tolist()
df, removed_outliers = remove_outliers(df, numeric_cols_all, outlier_method)
if removed_outliers > 0:
    st.write(
        f"✅ Outlier removal ({outlier_method}) removed {removed_outliers} rows.")

st.subheader("🧼 Cleaning Summary")
st.write(f"After cleaning: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(), use_container_width=True)

# Download cleaned dataset
csv_clean = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download cleaned dataset (CSV)",
    data=csv_clean,
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------
# Build pipeline and train
# ---------------------------------------------------------------
y = df[target_col]
X = df.drop(columns=[target_col])

pipe, num_cols, cat_cols = build_preprocessor(
    X,
    imputation_strategy=imputation_strategy,
    skew_strategy=skew_strategy,
    cat_encoding=cat_encoding,
    variance_threshold=variance_threshold_val if variance_threshold_val > 0 else None,
    rfe_n_features=rfe_n_features_val if rfe_n_features_val > 0 else None,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

with st.spinner("Training model and computing metrics..."):
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Pipeline training failed: {e}")
        st.stop()

y_pred = pipe.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
    "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
}

# ROC
roc_info = None
y_unique = y.unique()
if hasattr(pipe, "predict_proba") and len(y_unique) == 2:
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        roc_info = {"fpr": fpr, "tpr": tpr, "auc": float(auc_val)}
    except Exception:
        roc_info = None

st.success("✅ Model training complete.")

# ---------------------------------------------------------------
# Metrics + HTML report
# ---------------------------------------------------------------
st.subheader("📈 Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col2.metric("Precision (weighted)", f"{metrics['precision_weighted']:.3f}")
col3.metric("Recall (weighted)", f"{metrics['recall_weighted']:.3f}")
col4.metric("F1 (weighted)", f"{metrics['f1_weighted']:.3f}")

class_rep = classification_report(y_test, y_pred)
st.text("Classification report:")
st.text(class_rep)

config = {
    "Imputation": imputation_strategy,
    "Outlier removal": outlier_method,
    "Skew correction": skew_strategy,
    "Categorical encoding": cat_encoding,
    "VarianceThreshold": variance_threshold_val,
    "RFE n_features": rfe_n_features_val,
    "Test size": test_size,
    "Rows (cleaned)": len(df),
    "Columns (cleaned)": df.shape[1],
}
html_report = generate_html_report(config, metrics, class_rep)
st.download_button(
    label="⬇️ Download HTML model report",
    data=html_report,
    file_name="model_report.html",
    mime="text/html",
)

# ---------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
fig_cm.colorbar(im, ax=ax_cm)
st.pyplot(fig_cm)

# ---------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------
st.subheader("📉 ROC Curve")
if roc_info is not None:
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(roc_info["fpr"], roc_info["tpr"],
                label=f"AUC = {roc_info['auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)
else:
    st.info(
        "ROC curve is only shown for binary classification problems with predict_proba.")

# ---------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------
st.subheader("📊 Feature Importance (RandomForest)")

X_train_proc = pipe["pre"].transform(X_train)
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()

feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols:
    pre = pipe["pre"]
    cat_transformer = pre.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    if isinstance(encoder, OneHotEncoder):
        cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_feature_names)
    else:
        feature_names.extend(cat_cols)

importances = pipe["model"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(7, max(3, len(fi_df) * 0.3)))
ax_fi.barh(fi_df["feature"], fi_df["importance"])
ax_fi.invert_yaxis()
ax_fi.set_xlabel("Importance")
ax_fi.set_title("RandomForest Feature Importance")
st.pyplot(fig_fi)

# ---------------------------------------------------------------
# SHAP feature importance (top 10)
# ---------------------------------------------------------------
st.subheader("🧠 SHAP Feature Importance (Top 10)")

try:
    sample_size = min(300, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)

    X_sample_proc = pipe["pre"].transform(X_sample)
    if hasattr(X_sample_proc, "toarray"):
        X_sample_proc = X_sample_proc.toarray()

    X_sample_df = pd.DataFrame(X_sample_proc, columns=feature_names)

    explainer = shap.TreeExplainer(pipe["model"])
    shap_values = explainer.shap_values(X_sample_df)

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            sv = shap_values[1]
        else:
            sv_arr = np.array(shap_values)
            sv = sv_arr.mean(axis=0)
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-10:]
    top_features = [feature_names[i] for i in top_idx]
    X_top = X_sample_df[top_features]
    sv_top = sv[:, top_idx]

    fig_shap = plt.figure(figsize=(7, 5))
    shap.summary_plot(sv_top, X_top, feature_names=top_features, show=False)
    st.pyplot(fig_shap)

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# ---------------------------------------------------------------
# Partial Dependence Plots
# ---------------------------------------------------------------
st.subheader("📉 Partial Dependence Plots (PDP)")

numeric_cols_all = X.select_dtypes(include=["number"]).columns.tolist()
candidate_pdp_cols = [c for c in [
    "Age", "Fare", "Pclass"] if c in numeric_cols_all]
if not candidate_pdp_cols:
    candidate_pdp_cols = numeric_cols_all[:2]

if not candidate_pdp_cols:
    st.info("No numeric columns found for PDP.")
else:
    cols = st.columns(len(candidate_pdp_cols))
    for col_ax, feature in zip(cols, candidate_pdp_cols):
        with col_ax:
            try:
                pdp_res = partial_dependence(
                    pipe, X=X_train, features=[feature])
                grid = pdp_res.get("grid_values", pdp_res.get("values"))[0]
                avg = pdp_res["average"][0]

                fig_pdp, ax_pdp = plt.subplots()
                ax_pdp.plot(grid, avg)
                ax_pdp.set_xlabel(feature)
                ax_pdp.set_ylabel("Partial dependence")
                ax_pdp.set_title(f"PDP – {feature}")
                st.pyplot(fig_pdp)
            except Exception as e:
                st.warning(f"Could not compute PDP for {feature}: {e}")
