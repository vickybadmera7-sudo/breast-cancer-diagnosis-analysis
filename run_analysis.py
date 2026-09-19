"""
run_analysis.py - runs the full analysis once and saves the charts used in the README.

    python run_analysis.py

Outputs: images/*.png and model_results.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt

import pipeline as pl
import plots

ROOT = Path(__file__).parent
IMG = ROOT / "images"


def save(fig, name: str) -> None:
    IMG.mkdir(exist_ok=True)
    fig.savefig(IMG / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", IMG / name)


def main() -> None:
    exp = pl.run_experiment()
    df, feats = exp.df, exp.features

    print("Data quality:", exp.quality)

    # 1. EDA
    save(plots.fig_class_balance(df), "01_class_balance.png")
    top4 = list(exp.rf_importance.index[:4])
    save(plots.fig_distributions(df, top4), "02_top_feature_distributions.png")
    save(plots.fig_target_correlation(df, feats), "03_correlation_with_target.png")
    mean_cols = [c for c in feats if c.startswith("mean ")]
    save(plots.fig_heatmap(df, mean_cols), "04_mean_feature_heatmap.png")

    # 2. Modelling
    save(plots.fig_model_comparison(exp.table), "05_model_comparison.png")
    save(plots.fig_roc(exp.roc), "06_roc_curves.png")
    save(plots.fig_confusion(exp.cms[exp.best_name], f"{exp.best_name} - test set"),
         "07_confusion_matrix_best.png")
    save(plots.fig_importance(exp.rf_importance), "08_feature_importance.png")
    save(plots.fig_threshold(exp.y_test, exp.proba[exp.best_name]), "09_threshold_tradeoff.png")

    exp.table.to_csv(ROOT / "model_results.csv", index=False)
    print("saved", ROOT / "model_results.csv")
    print(exp.table.to_string(index=False))
    print("Best model (by cross-validated ROC-AUC):", exp.best_name)


if __name__ == "__main__":
    main()
