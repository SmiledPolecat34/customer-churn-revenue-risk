"""
Modèle de régression — Risque Revenus (Revenue at Risk).

Logique métier :
    revenue_at_risk = churn_probability × total_revenue

    Ce score estime, pour chaque client, le montant de revenus menacés
    en cas de départ. C'est la cible de régression que l'on cherche à prédire
    à partir des features comportementales et contractuelles.

Modèles :
    1. Ridge Regression (baseline linéaire régularisé)
    2. Random Forest Regressor
    3. XGBoost Regressor (gradient boosting)

Métriques : MAE, RMSE, R²
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

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBRegressor
import joblib

from src.config import (
    PREPROCESSOR_PATH, REVENUE_MODEL_PATH, METRICS_DIR,
    FIGURES_DIR, MODELS_DIR, RANDOM_STATE, CV_FOLDS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_REVENUE, TARGET_CHURN,
)
from src.utils import get_logger, save_model, ensure_dirs

logger = get_logger(__name__)


def compute_revenue_at_risk(
    y_churn_proba: np.ndarray,
    y_revenue: np.ndarray,
) -> np.ndarray:
    """
    Calcule le score métier 'revenue_at_risk' :
        revenue_at_risk = churn_probability × total_revenue
    """
    return y_churn_proba * y_revenue


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    """Calcule MAE, RMSE et R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    logger.info(f"  {model_name} → MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    return {
        "mae":  round(mae, 2),
        "rmse": round(rmse, 2),
        "r2":   round(r2, 4),
    }


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Scatter plot prédictions vs réelles."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color="steelblue")
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="Parfait")
    ax.set_xlabel("Revenue at Risk réel (€)"); ax.set_ylabel("Revenue at Risk prédit (€)")
    ax.set_title(f"Prédictions vs Réels — {model_name}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = model_name.replace(" ", "_").lower()
    fig.savefig(FIGURES_DIR / f"revenue_pred_vs_actual_{fname}.png", dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Distribution des résidus."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Résidus vs prédictions
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10, color="coral")
    axes[0].axhline(0, color="black", lw=1, linestyle="--")
    axes[0].set_xlabel("Prédit (€)"); axes[0].set_ylabel("Résidu (€)")
    axes[0].set_title(f"Résidus vs Prédictions — {model_name}", fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Distribution des résidus
    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Résidu (€)"); axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution des résidus", fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = model_name.replace(" ", "_").lower()
    fig.savefig(FIGURES_DIR / f"revenue_residuals_{fname}.png", dpi=150)
    plt.close(fig)


def plot_revenue_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """Bar chart comparatif MAE / RMSE / R²."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    for ax, metric, title in zip(
        axes,
        ["mae", "rmse", "r2"],
        ["MAE (€) ↓", "RMSE (€) ↓", "R² ↑"],
    ):
        vals = metrics_df.set_index("model")[metric]
        bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Comparaison des modèles — Revenue at Risk", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "revenue_models_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Comparaison revenue sauvegardée.")


def build_revenue_models(preprocessor) -> dict:
    """Construit les pipelines de régression."""
    return {
        "Ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=200, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="rmse", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0,
            )),
        ]),
    }


def train_revenue_model(
    X_train, X_test,
    y_train_churn_proba: np.ndarray,
    y_test_churn_proba: np.ndarray,
    y_train_revenue: pd.Series,
    y_test_revenue: pd.Series,
    preprocessor,
) -> tuple:
    """
    Entraîne et évalue les modèles de régression sur la cible 'revenue_at_risk'.

    Returns
    -------
    best_model, best_name, metrics_df
    """
    ensure_dirs()

    # Construction de la cible métier
    y_train_target = compute_revenue_at_risk(y_train_churn_proba, y_train_revenue.values)
    y_test_target  = compute_revenue_at_risk(y_test_churn_proba, y_test_revenue.values)

    logger.info(f"Revenue at Risk — distribution train : mean={y_train_target.mean():.2f}€ | "
                f"max={y_train_target.max():.2f}€ | std={y_train_target.std():.2f}€")

    models = build_revenue_models(preprocessor)
    results = {}

    for name, pipeline in models.items():
        logger.info(f"Entraînement régression : {name} ...")
        pipeline.fit(X_train, y_train_target)

        y_pred = pipeline.predict(X_test)
        metrics = compute_regression_metrics(y_test_target, y_pred, name)
        results[name] = {"model": pipeline, "pred": y_pred, "metrics": metrics}

        plot_predictions_vs_actual(y_test_target, y_pred, name)
        plot_residuals(y_test_target, y_pred, name)

        # Cross-validation (RMSE négatif → on prend la valeur absolue)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train_target, cv=cv,
                                    scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_rmse = -cv_scores
        metrics["cv_rmse_mean"] = round(cv_rmse.mean(), 2)
        metrics["cv_rmse_std"]  = round(cv_rmse.std(), 2)
        logger.info(f"  CV RMSE : {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

    # Tableau comparatif
    rows = [{"model": name, **res["metrics"]} for name, res in results.items()]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_DIR / "revenue_models_comparison.csv", index=False)
    logger.info(f"Métriques revenue sauvegardées.")

    plot_revenue_metrics_comparison(metrics_df)

    # Meilleur modèle : R² le plus élevé
    best_name = metrics_df.set_index("model")["r2"].idxmax()
    best_model = results[best_name]["model"]
    logger.info(f"\n✓ Meilleur modèle revenue : {best_name} "
                f"(R²={metrics_df.set_index('model').loc[best_name, 'r2']:.4f})")

    save_model(best_model, REVENUE_MODEL_PATH)
    for name, res in results.items():
        fname = "revenue_" + name.replace(" ", "_").lower() + ".joblib"
        save_model(res["model"], MODELS_DIR / fname)

    return best_model, best_name, metrics_df


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    from src.train_churn_models import train_and_evaluate

    X_train, X_test, y_train_churn, y_test_churn, y_train_rev, y_test_rev, preprocessor = run_preprocessing()
    best_churn, _, _ = train_and_evaluate(X_train, X_test, y_train_churn, y_test_churn, preprocessor)

    # On a besoin des probabilités du modèle churn pour construire la cible
    y_train_proba = best_churn.predict_proba(X_train)[:, 1]
    y_test_proba  = best_churn.predict_proba(X_test)[:, 1]

    train_revenue_model(
        X_train, X_test,
        y_train_proba, y_test_proba,
        y_train_rev, y_test_rev,
        preprocessor,
    )
