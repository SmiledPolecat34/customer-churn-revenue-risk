"""
Point d'entrée principal du pipeline Data Science.

Étapes exécutées :
    1. Prétraitement des données
    2. Entraînement des modèles de classification (churn)
    3. Entraînement du modèle de régression (revenue at risk)
    4. Calcul des valeurs SHAP et interprétabilité
    5. Génération d'un rapport de synthèse

Usage :
    python main.py                  # pipeline complet
    python main.py --skip-shap      # sans SHAP (plus rapide)
    python main.py --only-predict   # charge et teste les modèles
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Configuration du PATH ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, ensure_dirs
from src.config import METRICS_DIR, REPORTS_DIR, FIGURES_DIR

logger = get_logger("main")


def run_pipeline(skip_shap: bool = False) -> None:
    """Exécute le pipeline complet de Data Science."""
    start = time.time()
    ensure_dirs()

    logger.info("=" * 60)
    logger.info("PIPELINE DATA SCIENCE — Customer Churn & Revenue Risk")
    logger.info(f"Démarré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── ÉTAPE 1 : Prétraitement ───────────────────────────────────────────────
    logger.info("\n[1/4] PRÉTRAITEMENT DES DONNÉES ...")
    from src.preprocessing import run_preprocessing
    (X_train, X_test,
     y_train_churn, y_test_churn,
     y_train_rev, y_test_rev,
     preprocessor) = run_preprocessing()
    logger.info("  ✓ Prétraitement terminé.")

    # ── ÉTAPE 2 : Modèles de classification (churn) ───────────────────────────
    logger.info("\n[2/4] ENTRAÎNEMENT DES MODÈLES DE CLASSIFICATION ...")
    from src.train_churn_models import train_and_evaluate
    best_churn_model, best_churn_name, churn_metrics = train_and_evaluate(
        X_train, X_test,
        y_train_churn, y_test_churn,
        preprocessor,
    )
    logger.info(f"  ✓ Meilleur modèle churn : {best_churn_name}")

    # ── ÉTAPE 3 : Modèle de régression (revenue risk) ─────────────────────────
    logger.info("\n[3/4] ENTRAÎNEMENT DU MODÈLE REVENUE AT RISK ...")
    from src.train_revenue_model import train_revenue_model

    y_train_proba = best_churn_model.predict_proba(X_train)[:, 1]
    y_test_proba  = best_churn_model.predict_proba(X_test)[:, 1]

    best_rev_model, best_rev_name, rev_metrics = train_revenue_model(
        X_train, X_test,
        y_train_proba, y_test_proba,
        y_train_rev, y_test_rev,
        preprocessor,
    )
    logger.info(f"  ✓ Meilleur modèle revenue : {best_rev_name}")

    # ── ÉTAPE 4 : SHAP ────────────────────────────────────────────────────────
    if not skip_shap:
        logger.info("\n[4/4] CALCUL DES VALEURS SHAP ...")
        try:
            from src.explainability import explain_churn_model, plot_sklearn_feature_importance
            explain_churn_model(best_churn_model, X_test)
            plot_sklearn_feature_importance(best_churn_model)
            logger.info("  ✓ SHAP terminé.")
        except Exception as e:
            logger.warning(f"  ⚠ SHAP ignoré : {e}")
    else:
        logger.info("\n[4/4] SHAP ignoré (--skip-shap).")

    # ── Résumé ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info(f"Durée totale : {elapsed:.1f}s")

    # Meilleure métrique churn
    best_row = churn_metrics[churn_metrics["model"] == best_churn_name].iloc[0]
    logger.info(f"\nMeilleur modèle churn : {best_churn_name}")
    logger.info(f"  F1      = {best_row.get('f1', 'N/A')}")
    logger.info(f"  ROC-AUC = {best_row.get('roc_auc', 'N/A')}")
    logger.info(f"  PR-AUC  = {best_row.get('pr_auc', 'N/A')}")
    logger.info(f"  Recall  = {best_row.get('recall', 'N/A')}")

    # Meilleure métrique revenue
    best_rev_row = rev_metrics[rev_metrics["model"] == best_rev_name].iloc[0]
    logger.info(f"\nMeilleur modèle revenue : {best_rev_name}")
    logger.info(f"  MAE  = {best_rev_row.get('mae', 'N/A')} €")
    logger.info(f"  RMSE = {best_rev_row.get('rmse', 'N/A')} €")
    logger.info(f"  R²   = {best_rev_row.get('r2', 'N/A')}")

    logger.info(f"\nFigures sauvegardées → {FIGURES_DIR}")
    logger.info(f"Métriques sauvegardées → {METRICS_DIR}")
    logger.info("=" * 60)


def run_predict_test() -> None:
    """Teste le chargement des modèles et une prédiction sur un exemple."""
    logger.info("Test de prédiction ...")
    from src.predict import predict_full, load_models
    load_models()

    # Client exemple
    example = {
        "age": 35, "tenure_months": 6, "monthly_logins": 3,
        "weekly_active_days": 1, "avg_session_time": 8.0,
        "features_used": 2, "usage_growth_rate": -0.3,
        "last_login_days_ago": 25, "monthly_fee": 89.99,
        "payment_failures": 3, "support_tickets": 5,
        "avg_resolution_time": 72.0, "csat_score": 3.0,
        "escalations": 2, "email_open_rate": 0.05,
        "marketing_click_rate": 0.01, "nps_score": -3,
        "referral_count": 0,
        "gender": "Female", "customer_segment": "SMB",
        "signup_channel": "Online", "contract_type": "Monthly",
        "payment_method": "Credit Card", "discount_applied": "No",
        "price_increase_last_3m": "Yes", "complaint_type": "Billing",
        "survey_response": "Dissatisfied",
    }

    result = predict_full(example)
    logger.info("\nExemple client :")
    for k, v in result.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Data Science — Churn & Revenue Risk")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Ignorer le calcul SHAP (plus rapide)")
    parser.add_argument("--only-predict", action="store_true",
                        help="Tester uniquement la prédiction (modèles déjà entraînés)")
    args = parser.parse_args()

    if args.only_predict:
        run_predict_test()
    else:
        run_pipeline(skip_shap=args.skip_shap)
