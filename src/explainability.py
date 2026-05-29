"""
Interprétabilité des modèles — SHAP (SHapley Additive exPlanations).

Génère :
    - SHAP summary plot (beeswarm) — importance globale des features
    - SHAP bar plot — importance moyenne absolue
    - SHAP waterfall plot — explication individuelle
    - Feature importance sklearn (pour Random Forest)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import shap
import joblib

from src.config import (
    SHAP_DIR, FIGURES_DIR, CHURN_MODEL_PATH, PREPROCESSOR_PATH,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.utils import get_logger, load_model, ensure_dirs

logger = get_logger(__name__)


def get_feature_names(preprocessor) -> list:
    """
    Reconstruit la liste des noms de features après ColumnTransformer.
    """
    numeric_names = NUMERIC_FEATURES.copy()

    # Récupère les noms OHE pour les features catégorielles
    cat_pipeline = preprocessor.named_transformers_["cat"]
    encoder = cat_pipeline.named_steps["encoder"]
    cat_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))

    return numeric_names + cat_names


def explain_churn_model(
    pipeline,
    X_test: pd.DataFrame,
    n_samples: int = 500,
) -> None:
    """
    Calcule et sauvegarde les visualisations SHAP pour le modèle churn.

    Parameters
    ----------
    pipeline : ImbPipeline
        Pipeline entraîné (inclut preprocessor + smote + classifier)
    X_test : pd.DataFrame
        Données de test brutes (avant transformation)
    n_samples : int
        Nombre d'exemples utilisés pour l'explication (limité pour performance)
    """
    ensure_dirs()
    logger.info("Calcul des valeurs SHAP pour le modèle churn ...")

    # Extraction du preprocessor et du classifier depuis le pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    # Transformation des données
    X_transformed = preprocessor.transform(X_test)
    feature_names  = get_feature_names(preprocessor)

    # Sous-ensemble pour la vitesse
    if len(X_transformed) > n_samples:
        idx = np.random.RandomState(42).choice(len(X_transformed), n_samples, replace=False)
        X_shap = X_transformed[idx]
    else:
        X_shap = X_transformed

    X_shap_df = pd.DataFrame(X_shap, columns=feature_names)

    # Explainer SHAP
    classifier_type = type(classifier).__name__
    logger.info(f"  Classifier : {classifier_type}")

    try:
        if "XGB" in classifier_type:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap_df)
        elif "RandomForest" in classifier_type:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_shap_df)
            # Pour RF binaire, prend les SHAP de la classe positive
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Logistic Regression : LinearExplainer
            explainer = shap.LinearExplainer(classifier, X_shap_df)
            shap_values = explainer.shap_values(X_shap_df)
    except Exception as e:
        logger.warning(f"TreeExplainer échoué, utilisation de KernelExplainer : {e}")
        background = shap.sample(X_shap_df, 50, random_state=42)
        explainer = shap.KernelExplainer(
            lambda x: classifier.predict_proba(x)[:, 1], background
        )
        shap_values = explainer.shap_values(X_shap_df, nsamples=100)

    # ── 1. Summary plot (beeswarm) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_shap_df,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP — Importance des features (Churn)", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(SHAP_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP beeswarm sauvegardé.")

    # ── 2. Bar plot (importance moyenne |SHAP|) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_shap_df,
        plot_type="bar",
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP — Top 20 features par importance absolue (Churn)", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(SHAP_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP bar plot sauvegardé.")

    # ── 3. Sauvegarde des valeurs SHAP en CSV ─────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(SHAP_DIR / "shap_values_churn.csv", index=False)
    logger.info("Valeurs SHAP sauvegardées en CSV.")

    # ── 4. Top features summary ───────────────────────────────────────────────
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    top20 = mean_abs_shap.head(20)
    logger.info(f"\nTop 10 features SHAP (churn) :\n{top20.head(10).to_string()}")

    top20.to_csv(SHAP_DIR / "top_features_shap.csv", header=["mean_abs_shap"])
    logger.info("Top features SHAP sauvegardé.")

    return mean_abs_shap


def plot_sklearn_feature_importance(pipeline, top_n: int = 20) -> None:
    """
    Feature importance native sklearn pour Random Forest.
    Utilisée en complément de SHAP quand le modèle est RF.
    """
    classifier = pipeline.named_steps.get("classifier")
    if not hasattr(classifier, "feature_importances_"):
        logger.info("Le classifier n'a pas d'attribut feature_importances_. Ignoré.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    importances   = classifier.feature_importances_

    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top = imp_series.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    top.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="black", lw=0.4)
    ax.set_title(f"Feature Importance sklearn — Top {top_n}", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_sklearn.png", dpi=150)
    plt.close(fig)
    logger.info("Feature importance sklearn sauvegardée.")


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    from src.utils import load_model

    X_train, X_test, y_train_churn, y_test_churn, *_ = run_preprocessing()
    model = load_model(CHURN_MODEL_PATH)
    explain_churn_model(model, X_test)
    plot_sklearn_feature_importance(model)
