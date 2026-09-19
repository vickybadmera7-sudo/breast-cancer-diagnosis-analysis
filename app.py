"""
Breast Cancer Diagnosis - end-to-end healthcare data science project.
Run locally:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import pipeline as pl
import plots

st.set_page_config(page_title="Breast Cancer Diagnosis Analysis", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner="Training models (takes a few seconds the first time)...")
def get_experiment() -> pl.Experiment:
    return pl.run_experiment()


def show(fig) -> None:
    """Render a matplotlib figure, then free its memory."""
    st.pyplot(fig)
    plt.close(fig)


exp = get_experiment()
df, feats, q = exp.df, exp.features, exp.quality
best = exp.best_name
best_row = exp.table.iloc[0]

st.title("🩺 Breast Cancer Diagnosis Analysis")
st.caption("Healthcare data science project · Breast Cancer Wisconsin (Diagnostic) dataset")

tab_over, tab_eda, tab_models, tab_predict, tab_end = st.tabs(
    ["📋 Overview", "🔍 Explore", "🤖 Models", "🧪 Predict", "✅ Conclusions"]
)

# ---------------------------------------------------------------- Overview
with tab_over:
    st.subheader("The problem")
    st.write(
        "Early and accurate diagnosis of breast cancer saves lives. Doctors examine cells "
        "taken with a fine needle; the cell nuclei are measured from a digitised image. "
        "**Can those measurements alone tell a malignant tumour from a benign one?** "
        "This project analyses the data end to end and compares five machine-learning models."
    )
    c1, c2 = st.columns(2)
    c1.metric("Patients", q["rows"])
    c2.metric("Measurements each", q["features"])
    c3, c4 = st.columns(2)
    c3.metric("Malignant cases", f'{q["malignant"]} ({q["malignant_pct"]}%)')
    c4.metric("Missing values", q["missing_values"])

    st.subheader("Data preview")
    st.dataframe(df.head(10))
    st.caption(
        "Every feature is computed per cell nucleus: **mean**, **error** (standard error) and "
        "**worst** (mean of the three largest values) of radius, texture, perimeter, area, "
        "smoothness, compactness, concavity, concave points, symmetry and fractal dimension."
    )

    st.subheader("Method")
    st.markdown(
        "1. **Clean & check** the data (no missing values or duplicates)\n"
        "2. **Explore** distributions and correlations\n"
        "3. **Split** 80% train / 20% test (stratified)\n"
        "4. **Compare** 5 models with 5-fold cross-validation on the train set\n"
        "5. **Evaluate** once on the untouched test set\n"
        "6. **Explain** which measurements matter and how the threshold changes results"
    )
    st.info(
        "Educational project only. It is **not** a medical device and must not be used "
        "to diagnose anyone."
    )

# ---------------------------------------------------------------- Explore
with tab_eda:
    st.subheader("Class balance")
    show(plots.fig_class_balance(df))
    st.write(
        f"About {q['malignant_pct']}% of cases are malignant. The classes are moderately "
        "imbalanced, so accuracy alone is not enough - recall for malignant cases matters."
    )

    st.subheader("Feature explorer")
    default_idx = feats.index(exp.rf_importance.index[0])
    feature = st.selectbox("Pick a measurement", feats, index=default_idx)
    show(plots.fig_distributions(df, [feature], ncols=1))
    st.dataframe(pl.group_summary(df, feature))

    st.subheader("What is linked to malignancy?")
    show(plots.fig_target_correlation(df, feats))
    st.write(
        "Size and shape-irregularity measurements (perimeter, area, radius, concave points) "
        "show the strongest link. Malignant nuclei tend to be larger and have more "
        "irregular, indented outlines."
    )

    with st.expander("Correlation heatmap of the 'mean' features"):
        show(plots.fig_heatmap(df, [c for c in feats if c.startswith("mean ")]))
        st.caption(
            "Radius, perimeter and area are almost perfectly correlated (they describe "
            "the same thing: size). This is why importance is shared between them."
        )

# ---------------------------------------------------------------- Models
with tab_models:
    st.subheader("Model comparison")
    st.write(
        "Models are ranked by **cross-validated ROC-AUC** on the training data. "
        "The 'Test' columns come from the 20% of patients the models never saw."
    )
    st.dataframe(exp.table.set_index("Model"))
    show(plots.fig_model_comparison(exp.table))
    st.success(
        f"Best by cross-validation: **{best}** - test accuracy {best_row['Test Accuracy']:.1%}, "
        f"recall {best_row['Test Recall']:.1%}, ROC-AUC {best_row['Test ROC-AUC']:.3f}."
    )
    st.caption(
        "The test set has only 114 patients (42 malignant), so one patient changes "
        "recall by about 2.4 points. Top models are statistically very close."
    )

    st.subheader("ROC curves")
    show(plots.fig_roc(exp.roc))

    st.subheader("Confusion matrix")
    chosen = st.selectbox("Model", list(exp.models), index=list(exp.models).index(best))
    show(plots.fig_confusion(exp.cms[chosen], f"{chosen} - test set"))
    fn, fp = int(exp.cms[chosen][1, 0]), int(exp.cms[chosen][0, 1])
    st.write(f"**{fn}** cancer(s) missed and **{fp}** false alarm(s) on the test set.")

    st.subheader("Which measurements matter most?")
    show(plots.fig_importance(exp.rf_importance))

    st.subheader(f"Decision threshold ({best})")
    st.write(
        "By default a case is flagged when P(malignant) ≥ 0.50. In medicine, missing a "
        "cancer is usually worse than a false alarm, so try lowering the threshold."
    )
    threshold = st.slider("Threshold", 0.05, 0.95, 0.50, 0.01)
    m = pl.metrics_at_threshold(exp.y_test, exp.proba[best], threshold)
    a, b, c = st.columns(3)
    a.metric("Recall", f"{m['recall']:.1%}")
    b.metric("Precision", f"{m['precision']:.1%}")
    c.metric("Missed / False alarms", f"{m['fn']} / {m['fp']}")
    show(plots.fig_threshold(exp.y_test, exp.proba[best], threshold))

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Try the model")
    st.write(
        "A compact logistic-regression model that uses only "
        f"{len(exp.compact_features)} non-redundant measurements "
        f"(test accuracy {exp.compact_metrics['accuracy']:.1%}, "
        f"ROC-AUC {exp.compact_metrics['roc_auc']:.3f})."
    )
    preset = st.radio(
        "Start from",
        ["Typical patient (median values)", "A benign patient from the test set",
         "A malignant patient from the test set"],
    )
    cf = exp.compact_features
    if preset.startswith("A benign"):
        start = exp.X_test[exp.y_test == 0].iloc[0][cf]
    elif preset.startswith("A malignant"):
        start = exp.X_test[exp.y_test == 1].iloc[0][cf]
    else:
        start = df[cf].median()

    values = {}
    for feat in cf:
        lo, hi = float(df[feat].min()), float(df[feat].max())
        values[feat] = st.slider(
            feat, lo, hi, float(start[feat]), (hi - lo) / 200,
            format="%.4f" if hi < 1 else "%.2f", key=f"{preset}|{feat}",
        )

    p = pl.predict_malignant(exp, values)
    st.metric("Estimated probability of malignancy", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= 0.5:
        st.error("The model would flag this record as **malignant**.")
    else:
        st.success("The model would classify this record as **benign**.")
    st.caption("Educational demo only - not medical advice.")

# ---------------------------------------------------------------- Conclusions
with tab_end:
    top3 = list(exp.rf_importance.index[:3])
    st.subheader("Key findings")
    lines = []
    for f in top3[:2]:
        g = pl.group_summary(df, f)
        ratio = g.loc["Malignant", "mean"] / g.loc["Benign", "mean"]
        lines.append(
            f"- **{f}** averages {g.loc['Malignant', 'mean']:.2f} in malignant cases vs "
            f"{g.loc['Benign', 'mean']:.2f} in benign ones ({ratio:.1f}x higher)."
        )
    lines += [
        f"- The most important measurements are **{top3[0]}**, **{top3[1]}** and **{top3[2]}**.",
        f"- **{best}** had the best cross-validated ROC-AUC ({best_row['CV ROC-AUC']:.3f}); on the "
        f"test set it reached {best_row['Test Accuracy']:.1%} accuracy and "
        f"{best_row['Test Recall']:.1%} recall.",
        "- A single Decision Tree is clearly weaker than the other four models, which are close to each other.",
        "- Lowering the decision threshold trades a few false alarms for fewer missed cancers.",
    ]
    st.markdown("\n".join(lines))

    st.subheader("Limitations")
    st.markdown(
        "- Only 569 patients from one source; results may not generalise to other hospitals.\n"
        "- The test set is small, so differences of one patient are noise.\n"
        "- Features are measurements from images, not full clinical history.\n"
        "- A real system would need clinical validation and doctor oversight."
    )

    st.subheader("Next steps")
    st.markdown(
        "- Hyper-parameter tuning and repeated cross-validation\n"
        "- Model explanations with SHAP\n"
        "- Validate on an independent external dataset"
    )
