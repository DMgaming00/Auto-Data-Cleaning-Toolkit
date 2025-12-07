import os
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

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
)
from sklearn.inspection import partial_dependence

import shap


def load_titanic() -> pd.DataFrame:
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


def remove_outliers(df: pd.DataFrame, numeric_cols: List[str], method: str):
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
            preds = iso.predict(non_na)
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
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if imputation_strategy == "Mean":
        num_imputer = SimpleImputer(strategy="mean")
    elif imputation_strategy == "Median":
        num_imputer = SimpleImputer(strategy="median")
    elif imputation_strategy == "KNN":
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        num_imputer = IterativeImputer(random_state=42)

    cat_imputer = SimpleImputer(strategy="most_frequent")

    skew_step = None
    if skew_strategy == "Log1p":
        skew_step = ("skew", FunctionTransformer(
            lambda x: np.log1p(np.clip(x, a_min=0, a_max=None))))
    elif skew_strategy == "Yeo-Johnson":
        skew_step = ("skew", PowerTransformer(method="yeo-johnson"))

    if cat_encoding == "One-Hot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
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


def run_experiment(
    target_col: str = "Survived",
    imputation_strategy: str = "Median",
    outlier_method: str = "IQR",
    skew_strategy: str = "Yeo-Johnson",
    cat_encoding: str = "One-Hot",
    variance_threshold: float = 0.0,
    rfe_n_features: int = 0,
    test_size: float = 0.2,
):
    df = load_titanic()

    df = df.dropna(subset=[target_col]).copy()
    df = expand_datetime_features(df)

    numeric_cols_all = df.drop(columns=[target_col]).select_dtypes(
        include=["number"]).columns.tolist()
    df, removed_outliers = remove_outliers(
        df, numeric_cols_all, outlier_method)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    pipe, num_cols, cat_cols = build_preprocessor(
        X,
        imputation_strategy=imputation_strategy,
        skew_strategy=skew_strategy,
        cat_encoding=cat_encoding,
        variance_threshold=variance_threshold if variance_threshold > 0 else None,
        rfe_n_features=rfe_n_features if rfe_n_features > 0 else None,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    with mlflow.start_run():
        params = {
            "target_col": target_col,
            "imputation_strategy": imputation_strategy,
            "outlier_method": outlier_method,
            "skew_strategy": skew_strategy,
            "cat_encoding": cat_encoding,
            "variance_threshold": variance_threshold,
            "rfe_n_features": rfe_n_features,
            "test_size": test_size,
        }
        mlflow.log_params(params)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        y_unique = y.unique()
        roc_info = None
        if hasattr(pipe, "predict_proba") and len(y_unique) == 2:
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_val = roc_auc_score(y_test, y_proba)
                roc_info = {"fpr": fpr, "tpr": tpr, "auc": float(auc_val)}
                metrics["roc_auc"] = float(auc_val)
            except Exception:
                roc_info = None

        mlflow.log_metrics(metrics)

        os.makedirs("artifacts", exist_ok=True)

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center",
                           va="center", color="black")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        fig_cm.colorbar(im, ax=ax_cm)
        cm_path = os.path.join("artifacts", "confusion_matrix.png")
        fig_cm.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="figures")

        # ROC curve artifact
        if roc_info is not None:
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(roc_info["fpr"], roc_info["tpr"],
                        label=f"AUC = {roc_info['auc']:.3f}")
            ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            roc_path = os.path.join("artifacts", "roc_curve.png")
            fig_roc.savefig(roc_path, bbox_inches="tight")
            plt.close(fig_roc)
            mlflow.log_artifact(roc_path, artifact_path="figures")

        # Feature names for SHAP / PDP
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
                cat_feature_names = encoder.get_feature_names_out(
                    cat_cols).tolist()
                feature_names.extend(cat_feature_names)
            else:
                feature_names.extend(cat_cols)

        # SHAP beeswarm (top 10)
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
            shap.summary_plot(
                sv_top, X_top, feature_names=top_features, show=False)
            shap_path = os.path.join("artifacts", "shap_beeswarm_top10.png")
            fig_shap.savefig(shap_path, bbox_inches="tight")
            plt.close(fig_shap)
            mlflow.log_artifact(shap_path, artifact_path="figures")

        except Exception as e:
            print("SHAP logging failed:", e)

        # PDP plots
        numeric_cols_all = X.select_dtypes(include=["number"]).columns.tolist()
        candidate_pdp_cols = [c for c in [
            "Age", "Fare", "Pclass"] if c in numeric_cols_all]
        if not candidate_pdp_cols:
            candidate_pdp_cols = numeric_cols_all[:2]

        for feature in candidate_pdp_cols:
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
                pdp_path = os.path.join("artifacts", f"pdp_{feature}.png")
                fig_pdp.savefig(pdp_path, bbox_inches="tight")
                plt.close(fig_pdp)
                mlflow.log_artifact(pdp_path, artifact_path="figures")
            except Exception as e:
                print(f"PDP failed for {feature}: {e}")

        # Log full pipeline as MLflow model
        mlflow.sklearn.log_model(pipe, "model")
        print("Run complete. Metrics:", metrics)


if __name__ == "__main__":
    run_experiment()
