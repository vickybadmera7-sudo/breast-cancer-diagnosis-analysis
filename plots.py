"""
plots.py - every chart used by the dashboard and the README.
Each function returns a matplotlib Figure (nothing is written to disk here).
"""
import matplotlib

matplotlib.use("Agg")  # headless backend: works on Streamlit Cloud and in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline import TARGET, metrics_at_threshold

PALETTE = {"Benign": "#2A9D8F", "Malignant": "#E76F51"}
sns.set_theme(style="whitegrid", context="notebook")


def _short(name: str) -> str:
    return name.replace("mean ", "").replace("worst ", "worst ")


def fig_class_balance(df: pd.DataFrame):
    counts = df["diagnosis"].value_counts().reindex(["Benign", "Malignant"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 4,
                f"{val}  ({100 * val / counts.sum():.1f}%)", ha="center", fontsize=10)
    ax.set_ylabel("Patients")
    ax.set_title("Diagnosis balance")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    return fig


def fig_distributions(df: pd.DataFrame, features, ncols: int = 2):
    features = list(features)
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(5.5, 3.9 * ncols), 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, feat) in enumerate(zip(axes, features)):
        sns.histplot(data=df, x=feat, hue="diagnosis", hue_order=["Benign", "Malignant"],
                     palette=PALETTE, kde=True, stat="density", common_norm=False,
                     element="step", alpha=0.35, ax=ax, legend=(i == 0))
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_target_correlation(df: pd.DataFrame, features, top: int = 10):
    corr = df[list(features)].corrwith(df[TARGET])
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(corr.index, corr.values, color="#E76F51")
    ax.set_xlabel("Correlation with malignant diagnosis")
    ax.set_title(f"Top {top} features linked to malignancy")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def fig_heatmap(df: pd.DataFrame, cols):
    cols = list(cols)
    corr = df[cols].corr()
    labels = [_short(c) for c in cols]
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between the 'mean' features")
    fig.tight_layout()
    return fig


def fig_model_comparison(table: pd.DataFrame):
    cols = ["Test Accuracy", "Test Recall", "Test ROC-AUC"]
    data = table.set_index("Model")[cols].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    data.plot.barh(ax=ax, width=0.78, color=["#264653", "#E76F51", "#2A9D8F"])
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Score on held-out test set (axis starts at 0.75)")
    ax.set_ylabel("")
    ax.set_title("Model comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_roc(roc: dict):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, (fpr, tpr, auc) in roc.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_confusion(cm, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14},
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def fig_importance(importance: pd.Series, top: int = 10, title: str = "Random Forest feature importance"):
    imp = importance.sort_values(ascending=False)[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color="#264653")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_threshold(y_true, proba, current: float = 0.5):
    ts = np.linspace(0.05, 0.95, 91)
    rows = [metrics_at_threshold(y_true, proba, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [r["recall"] for r in rows], label="Recall (cancers caught)", color="#E76F51", lw=2)
    ax.plot(ts, [r["precision"] for r in rows], label="Precision", color="#264653", lw=2)
    ax.plot(ts, [r["f1"] for r in rows], label="F1", color="#2A9D8F", lw=2, ls="--")
    ax.axvline(current, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Decision threshold for flagging 'malignant'")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Recall vs precision trade-off")
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    return fig
