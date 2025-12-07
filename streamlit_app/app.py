import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from io import StringIO
from typing import Optional, Dict, List

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

# Reduce DPI for Streamlit Cloud safety
plt.rcParams["figure.dpi"] = 72


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="CMPE 255 – Auto Data Toolkit (Enhanced)",
    layout="wide",
    page_icon="🚢",
)

st.title("🚢 CMPE 255 – Auto Data Toolkit (Enhanced)")
st.markdown(
    """
This app demonstrates an end-to-end **CRISP-DM** workflow with the features you
requested:

- Multiple imputation strategies  
- Outlier removal (IQR / IsolationForest)  
- Skew correction (log1p / Yeo-Johnson)  
- Categorical encoders (One-Hot / Ordinal)  
- Datetime feature engineering  
- Duplicate removal  
- Feature selection (VarianceThreshold / RFE)  
- RandomForest feature importance  
- Partial Dependence Plots (PDP)  
- Downloadable cleaned dataset + HTML report  

🔥 **High-cardinality text columns (Name, Ticket, Cabin) are automatically removed**  
to avoid massive one-hot encoding expansion and Streamlit display errors.
"""
)


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
@st.cache_data
def load_titanic_demo() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df


def expand_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        dt = None
        if np.issubdtype(s.dtype, np.datetime64):
            dt = s
        elif s.dtype == "object":
            try:
                dt_candidate = pd.to_datetime(s, errors="raise")
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


def remove_outliers(df: pd.DataFrame, numeric_cols: List[str], method: str):
    if method == "None" or not numeric_cols:
        return df, 0

    df = df.copy()
    initial_rows = len(df)

    if method == "IQR":
        mask = pd.Series(True, index=df.index)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask &= df[col].between(lower, upper) | df[col].isna()
        df = df[mask]

    elif method == "IsolationForest":
        numeric_df = df[numeric_cols].dropna()
        if len(numeric_df) > 0:
            iso = IsolationForest(random_state=42, contamination="auto")
            iso.fit(numeric_df)
            preds = iso.predict(numeric_df)
            keep_idx = numeric_df.index[preds == 1]
            df = df.loc[keep_idx]

    removed = initial_rows - len(df)
    return df, removed


# -------------------------------------------------------------------
# Preprocessor builder
# -------------------------------------------------------------------
def build_preprocessor(
    X: pd.DataFrame,
    imputation_strategy: str,
    skew_strategy: str,
    cat_encoding: str,
    variance_threshold: Optional[float],
    rfe_n_features: Optional[int],
):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Numeric imputers
    if imputation_strategy == "Mean":
        num_imputer = SimpleImputer(strategy="mean")
    elif imputation_strategy == "Median":
        num_imputer = SimpleImputer(strategy="median")
    elif imputation_strategy == "KNN":
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        num_imputer = IterativeImputer(random_state=42)

    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Skew correction
    skew_step = None
    if skew_strategy == "Log1p":
        skew_step = ("skew", FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None))))
    elif skew_strategy == "Yeo-Johnson":
        skew_step = ("skew", PowerTransformer(method="yeo-johnson"))

    # Categorical encoding
    if cat_encoding == "One-Hot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    num_steps = [("imputer", num_imputer)]
    if skew_step:
        num_steps.append(skew_step)
    num_steps.append(("scaler", StandardScaler()))
    num_tf = Pipeline(num_steps)

    cat_tf = Pipeline([
        ("imputer", cat_imputer),
        ("encoder", encoder),
    ])

    pre = ColumnTransformer([
        ("num", num_tf, numeric_cols),
        ("cat", cat_tf, categorical_cols),
    ])

    steps = [("pre", pre)]

    if variance_threshold and variance_threshold > 0:
        steps.append(("var_sel", VarianceThreshold(threshold=variance_threshold)))

    if rfe_n_features and rfe_n_features > 0:
        steps.append((
            "rfe",
            RFE(
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                ),
                n_features_to_select=rfe_n_features
            )
        ))

    steps.append(("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )))

    pipe = Pipeline(steps)
    return pipe, numeric_cols, categorical_cols


# -------------------------------------------------------------------
# HTML report generator
# -------------------------------------------------------------------
def generate_html_report(config: Dict[str, object], metrics: Dict[str, float], class_rep: str) -> str:
    html = StringIO()
    html.write("<html><head><title>Model Report</title></head><body>")
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
    html.write(class_rep)
    html.write("</pre></body></html>")
    return html.getvalue()


# -------------------------------------------------------------------
# Safe feature-name extractor
# -------------------------------------------------------------------
def get_feature_names_after_selection(pipe: Pipeline, numeric_cols: List[str], cat_cols: List[str]) -> List[str]:
    pre: ColumnTransformer = pipe.named_steps["pre"]

    feature_names = []
    feature_names.extend(numeric_cols)

    if cat_cols:
        encoder = pre.named_transformers_["cat"].named_steps["encoder"]
        if isinstance(encoder, OneHotEncoder):
            feature_names.extend(encoder.get_feature_names_out(cat_cols).tolist())
        else:
            feature_names.extend(cat_cols)

    if "var_sel" in pipe.named_steps:
        mask = pipe.named_steps["var_sel"].get_support()
        feature_names = [n for n, keep in zip(feature_names, mask) if keep]

    if "rfe" in pipe.named_steps:
        mask = pipe.named_steps["rfe"].get_support()
        feature_names = [n for n, keep in zip(feature_names, mask) if keep]

    return feature_names


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose dataset:", ["Titanic demo", "Upload CSV"], index=0)

if data_source == "Titanic demo":
    raw_df = load_titanic_demo()
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.stop()
    raw_df = pd.read_csv(uploaded)

# 🔥 Drop problematic high-cardinality columns
raw_df = raw_df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

st.subheader("📊 Raw Data Preview")
st.write(f"Shape: **{raw_df.shape[0]} rows × {raw_df.shape[1]} columns**")
st.dataframe(raw_df.head(), width="stretch")


# -------------------------------------------------------------------
# Sidebar: Data cleaning
# -------------------------------------------------------------------
st.sidebar.header("2. Data Cleaning & Preparation")

remove_dupes = st.sidebar.checkbox("Remove duplicate rows", value=True)

imputation_strategy = st.sidebar.selectbox("Numeric imputation strategy",
    ["Mean", "Median", "KNN", "Iterative"])

outlier_method = st.sidebar.selectbox("Outlier removal",
    ["None", "IQR", "IsolationForest"])

skew_strategy = st.sidebar.selectbox("Skew correction", ["None", "Log1p", "Yeo-Johnson"])
cat_encoding = st.sidebar.selectbox("Categorical encoding", ["One-Hot", "Ordinal"])

variance_threshold_val = st.sidebar.slider(
    "VarianceThreshold (0 disables)", 0.0, 0.2, 0.0, 0.01
)

rfe_n_features_val = st.sidebar.slider(
    "RFE: number of features (0 disables)", 0, 50, 0, 1
)

st.sidebar.header("3. Target & Split")

default_target = "Survived" if "Survived" in raw_df.columns else raw_df.columns[-1]
target_col = st.sidebar.selectbox("Target column", raw_df.columns, index=raw_df.columns.get_loc(default_target))

test_size = st.sidebar.slider("Test size fraction", 0.1, 0.4, 0.2, 0.05)

run_button = st.sidebar.button("🚀 Run Auto-Toolkit")

if not run_button:
    st.stop()


# -------------------------------------------------------------------
# Perform cleaning
# -------------------------------------------------------------------
df = raw_df.copy()

df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

if remove_dupes:
    df = df.drop_duplicates()

df = df.dropna(subset=[target_col]).copy()

df = expand_datetime_features(df)

num_cols_all = df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns.tolist()
df, removed_outliers = remove_outliers(df, num_cols_all, outlier_method)

st.subheader("🧼 Cleaning Summary")
st.write(f"After cleaning: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(), width="stretch")

st.download_button(
    "⬇️ Download cleaned dataset",
    df.to_csv(index=False).encode("utf-8"),
    "cleaned.csv",
    "text/csv"
)


# -------------------------------------------------------------------
# Model training
# -------------------------------------------------------------------
y = df[target_col]
X = df.drop(columns=[target_col])

pipe, numeric_cols, categorical_cols = build_preprocessor(
    X,
    imputation_strategy,
    skew_strategy,
    cat_encoding,
    variance_threshold_val if variance_threshold_val > 0 else None,
    rfe_n_features_val if rfe_n_features_val > 0 else None,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
}


st.subheader("📈 Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col2.metric("Precision (weighted)", f"{metrics['precision_weighted']:.3f}")
col3.metric("Recall (weighted)", f"{metrics['recall_weighted']:.3f}")
col4.metric("F1 (weighted)", f"{metrics['f1_weighted']:.3f}")

class_rep = classification_report(y_test, y_pred)
st.text(class_rep)


st.download_button(
    "⬇️ Download HTML model report",
    generate_html_report(
        {
            "Imputation": imputation_strategy,
            "Outlier removal": outlier_method,
            "Skew correction": skew_strategy,
            "Categorical encoding": cat_encoding,
            "VarianceThreshold": variance_threshold_val,
            "RFE n_features": rfe_n_features_val,
            "Test size": test_size,
            "Rows cleaned": len(df),
        },
        metrics,
        class_rep
    ),
    "model_report.html"
)


# -------------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------------
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i][j], ha="center", va="center")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)


# -------------------------------------------------------------------
# ROC Curve
# -------------------------------------------------------------------
st.subheader("📉 ROC Curve")

roc_info = None
if hasattr(pipe, "predict_proba") and len(y.unique()) == 2:
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        ax.plot([0,1],[0,1],"k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
    except:
        st.info("ROC unavailable.")
else:
    st.info("ROC only available for binary classification.")


# -------------------------------------------------------------------
# Feature Importance
# -------------------------------------------------------------------
st.subheader("📊 Feature Importance (RandomForest)")

try:
    feature_names = get_feature_names_after_selection(pipe, numeric_cols, categorical_cols)
    importances = pipe["model"].feature_importances_

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(20)  # Limit display size

    fig_fi, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["feature"], fi_df["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances")
    st.pyplot(fig_fi)

except Exception as e:
    st.warning(f"Feature importance failed: {e}")


# -------------------------------------------------------------------
# PDP Plots
# -------------------------------------------------------------------
st.subheader("📉 Partial Dependence Plots (PDP)")

numeric_cols_all = X.select_dtypes(include=["number"]).columns.tolist()
candidate_pdp_cols = [c for c in ["Age", "Fare", "Pclass"] if c in numeric_cols_all]
if not candidate_pdp_cols:
    candidate_pdp_cols = numeric_cols_all[:2]

# Convert integers to floats for PDP safety
X_pdp = X_train.copy()
for c in candidate_pdp_cols:
    if np.issubdtype(X_pdp[c].dtype, np.integer):
        X_pdp[c] = X_pdp[c].astype(float)

cols = st.columns(len(candidate_pdp_cols))

for col_ax, feature in zip(cols, candidate_pdp_cols):
    with col_ax:
        try:
            pdp_res = partial_dependence(pipe, X=X_pdp, features=[feature])
            grid = pdp_res["grid_values"][0]
            avg = pdp_res["average"][0]

            fig, ax = plt.subplots()
            ax.plot(grid, avg)
            ax.set_title(f"PDP – {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Partial dependence")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"PDP failed for {feature}: {e}")


