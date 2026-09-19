"""
pipeline.py - data loading, modelling and evaluation for the project.

Everything the dashboard (app.py) and the report script (run_analysis.py)
need lives here, so both always show exactly the same numbers.

Dataset: Breast Cancer Wisconsin (Diagnostic) - 569 real patient records.
Each record describes a cell nucleus from a fine-needle-aspirate image
(radius, texture, perimeter, area, ...). It ships inside scikit-learn, so
there is no CSV to download and no file-path problems on Streamlit Cloud.

Target: `malignant` (1 = malignant, 0 = benign).
Malignant is the "positive" class, so recall = share of cancers we catch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "malignant"
LABELS = {0: "Benign", 1: "Malignant"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Return the dataset as a DataFrame with `malignant` (0/1) and `diagnosis`."""
    frame = load_breast_cancer(as_frame=True).frame.copy()
    # scikit-learn encodes malignant as 0; flip it so malignant = 1 (positive class)
    frame[TARGET] = 1 - frame.pop("target")
    frame["diagnosis"] = frame[TARGET].map(LABELS)
    return frame


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (TARGET, "diagnosis")]


def data_quality(df: pd.DataFrame) -> dict:
    counts = df["diagnosis"].value_counts()
    return {
        "rows": int(len(df)),
        "features": len(feature_columns(df)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "benign": int(counts.get("Benign", 0)),
        "malignant": int(counts.get("Malignant", 0)),
        "malignant_pct": round(float(100 * counts.get("Malignant", 0) / len(df)), 1),
    }


def group_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Mean / median / std of one feature for benign vs malignant."""
    out = df.groupby("diagnosis")[feature].agg(["mean", "median", "std"]).round(3)
    return out.loc[["Benign", "Malignant"]]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=random_state)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def metrics_at_threshold(y_true, proba, threshold: float = 0.5) -> dict:
    """Confusion-matrix based metrics when we flag 'malignant' at `threshold`."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def select_compact_features(X_train: pd.DataFrame, importance: pd.Series,
                            n: int = 6, max_corr: float = 0.9) -> list:
    """Top-n most important features, skipping near-duplicates.

    Radius, perimeter and area are almost perfectly correlated, so a form with
    all three would let users enter impossible combinations. Walk down the
    importance ranking and keep a feature only if it is not >max_corr
    correlated with one already chosen (train data only, so no leakage).
    """
    corr = X_train.corr().abs()
    chosen: list = []
    for feat in importance.index:
        if all(corr.loc[feat, c] <= max_corr for c in chosen):
            chosen.append(feat)
        if len(chosen) == n:
            break
    return chosen


@dataclass
class Experiment:
    df: pd.DataFrame
    features: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    models: dict            # name -> fitted model
    table: pd.DataFrame     # comparison table (one row per model)
    roc: dict               # name -> (fpr, tpr, auc)
    cms: dict               # name -> 2x2 confusion matrix (rows = actual)
    proba: dict             # name -> P(malignant) on the test set
    best_name: str
    rf_importance: pd.Series
    compact_features: list
    compact_model: Pipeline
    compact_metrics: dict
    quality: dict


def run_experiment(random_state: int = RANDOM_STATE, n_compact: int = 6) -> Experiment:
    df = load_data()
    features = feature_columns(df)
    X, y = df[features], df[TARGET]

    # 80/20 split, stratified so both sets keep the benign/malignant ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rows, fitted, roc, cms, proba = [], {}, {}, {}, {}
    for name, model in build_models(random_state).items():
        # model choice is based on cross-validation on the TRAIN set only
        cvr = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "recall": "recall", "roc_auc": "roc_auc"},
        )
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        m = metrics_at_threshold(y_test, p, 0.5)
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)

        fitted[name], proba[name], roc[name] = model, p, (fpr, tpr, auc)
        cms[name] = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        rows.append({
            "Model": name,
            "CV Accuracy": cvr["test_accuracy"].mean(),
            "CV Recall": cvr["test_recall"].mean(),
            "CV ROC-AUC": cvr["test_roc_auc"].mean(),
            "Test Accuracy": m["accuracy"],
            "Test Precision": m["precision"],
            "Test Recall": m["recall"],
            "Test F1": m["f1"],
            "Test ROC-AUC": auc,
            "Missed Cancers (FN)": m["fn"],
            "False Alarms (FP)": m["fp"],
        })

    table = (
        pd.DataFrame(rows)
        .sort_values(["CV ROC-AUC", "CV Recall"], ascending=False)
        .reset_index(drop=True)
    )
    num_cols = [c for c in table.columns if c not in ("Model", "Missed Cancers (FN)", "False Alarms (FP)")]
    table[num_cols] = table[num_cols].round(4)
    best_name = table.loc[0, "Model"]

    # Feature importance from the random forest (impurity based, train set only)
    rf_importance = (
        pd.Series(fitted["Random Forest"].feature_importances_, index=features)
        .sort_values(ascending=False)
    )

    # Compact model on a few non-redundant top features -> usable as an interactive form
    compact_features = select_compact_features(X_train, rf_importance, n=n_compact)
    compact_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
    ).fit(X_train[compact_features], y_train)
    cp = compact_model.predict_proba(X_test[compact_features])[:, 1]
    compact_metrics = metrics_at_threshold(y_test, cp, 0.5)
    compact_metrics["roc_auc"] = roc_auc_score(y_test, cp)

    return Experiment(
        df=df, features=features,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        models=fitted, table=table, roc=roc, cms=cms, proba=proba,
        best_name=best_name, rf_importance=rf_importance,
        compact_features=compact_features, compact_model=compact_model,
        compact_metrics=compact_metrics, quality=data_quality(df),
    )


def predict_malignant(exp: Experiment, values: dict) -> float:
    """P(malignant) from the compact model given {feature: value}."""
    row = pd.DataFrame([values])[exp.compact_features]
    return float(exp.compact_model.predict_proba(row)[0, 1])
