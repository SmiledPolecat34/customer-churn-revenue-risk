"""
Entraînement et comparaison des modèles de classification (churn).

Modèles :
    1. Logistic Regression (baseline interprétable)
    2. Random Forest (non-linéaire, robuste)
    3. XGBoost (gradient boosting, souvent le meilleur)

Gestion du déséquilibre :
    - class_weight='balanced'
    - SMOTE sur train set
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib

from src.config import (
    PREPROCESSOR_PATH, CHURN_MODEL_PATH, METRICS_CSV,
    FIGURES_DIR, MODELS_DIR, RANDOM_STATE, CV_FOLDS, CHURN_THRESHOLD,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_CHURN,
)
from src.utils import get_logger, save_model, ensure_dirs

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calcule toutes les métriques d'évaluation."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc":    round(average_precision_score(y_true, y_proba), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_title(f"Matrice de confusion — {model_name}", fontweight="bold")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(results: dict, y_test: np.ndarray) -> None:
    """Trace et sauvegarde les courbes ROC de tous les modèles."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux de Faux Positifs"); ax.set_ylabel("Taux de Vrais Positifs")
    ax.set_title("Courbes ROC — Comparaison des modèles", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Courbes ROC sauvegardées.")


def plot_pr_curves(results: dict, y_test: np.ndarray) -> None:
    """Trace et sauvegarde les courbes Precision-Recall."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        ax.plot(rec, prec, label=f"{name} (PR-AUC={res['metrics']['pr_auc']:.3f})", lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall — Comparaison", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Courbes PR sauvegardées.")


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """Bar chart comparatif des métriques."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.set_index("model")[["f1", "roc_auc", "pr_auc", "recall", "precision"]].plot(
        kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5
    )
    ax.set_title("Comparaison des modèles — métriques principales", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=0); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "models_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Comparaison des modèles sauvegardée.")


def build_models(preprocessor) -> dict:
    """
    Construit les pipelines de classification.
    Retourne un dict {nom: pipeline}.
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    models = {
        "Logistic Regression": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                C=1.0, solver="lbfgs", random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("classifier", XGBClassifier(
                n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=9,
                eval_metric="logloss", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0,
            )),
        ]),
    }
    return models


def train_and_evaluate(
    X_train, X_test,
    y_train: pd.Series, y_test: pd.Series,
    preprocessor,
) -> tuple:
    """
    Entraîne, évalue et compare tous les modèles.

    Returns
    -------
    best_model, best_name, metrics_df
    """
    ensure_dirs()
    models = build_models(preprocessor)
    results = {}

    for name, pipeline in models.items():
        logger.info(f"Entraînement : {name} ...")
        pipeline.fit(X_train, y_train)

        # Prédictions
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= CHURN_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test.values, y_pred, y_proba)
        results[name] = {"model": pipeline, "proba": y_proba, "pred": y_pred, "metrics": metrics}

        logger.info(f"  {name} → F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | Recall={metrics['recall']:.4f}")
        plot_confusion_matrix(y_test.values, y_pred, name)

        # Validation croisée (F1 macro)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"]  = round(cv_scores.std(), 4)
        logger.info(f"  CV F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Graphiques comparatifs
    plot_roc_curves(results, y_test.values)
    plot_pr_curves(results, y_test.values)

    # Tableau comparatif
    rows = [{"model": name, **res["metrics"]} for name, res in results.items()]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    logger.info(f"Métriques sauvegardées → {METRICS_CSV}")

    plot_metrics_comparison(metrics_df)

    # Sélection du meilleur modèle (PR-AUC prioritaire sur dataset déséquilibré)
    best_name = metrics_df.set_index("model")["pr_auc"].idxmax()
    best_model = results[best_name]["model"]
    logger.info(f"\n✓ Meilleur modèle : {best_name} (PR-AUC={metrics_df.set_index('model').loc[best_name, 'pr_auc']:.4f})")

    # Sauvegarde
    save_model(best_model, CHURN_MODEL_PATH)
    # Sauvegarde tous les modèles
    for name, res in results.items():
        fname = name.replace(" ", "_").lower() + ".joblib"
        save_model(res["model"], MODELS_DIR / fname)

    return best_model, best_name, metrics_df


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    X_train, X_test, y_train_churn, y_test_churn, _, _, preprocessor = run_preprocessing()
    train_and_evaluate(X_train, X_test, y_train_churn, y_test_churn, preprocessor)
