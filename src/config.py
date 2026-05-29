"""
Configuration centrale du projet.
Tous les chemins, paramètres et constantes sont définis ici.
"""

from pathlib import Path

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"
METRICS_DIR    = REPORTS_DIR / "metrics"
SHAP_DIR       = REPORTS_DIR / "shap"

# Fichiers principaux
RAW_CSV             = DATA_RAW / "customer_churn.csv"
PROCESSED_TRAIN     = DATA_PROCESSED / "train.parquet"
PROCESSED_TEST      = DATA_PROCESSED / "test.parquet"
PREPROCESSOR_PATH   = MODELS_DIR / "preprocessor.joblib"
CHURN_MODEL_PATH    = MODELS_DIR / "best_churn_model.joblib"
REVENUE_MODEL_PATH  = MODELS_DIR / "revenue_risk_model.joblib"
METRICS_CSV         = METRICS_DIR / "models_comparison.csv"

# ── Dataset ──────────────────────────────────────────────────────────────────
TARGET_CHURN    = "churn"
TARGET_REVENUE  = "total_revenue"
ID_COL          = "customer_id"

# Colonnes à exclure du feature set
DROP_COLS = [ID_COL, TARGET_CHURN, TARGET_REVENUE, "city", "country"]

# Variables numériques
NUMERIC_FEATURES = [
    "age", "tenure_months", "monthly_logins", "weekly_active_days",
    "avg_session_time", "features_used", "usage_growth_rate",
    "last_login_days_ago", "monthly_fee", "payment_failures",
    "support_tickets", "avg_resolution_time", "csat_score",
    "escalations", "email_open_rate", "marketing_click_rate",
    "nps_score", "referral_count",
]

# Variables catégorielles
CATEGORICAL_FEATURES = [
    "gender", "customer_segment", "signup_channel", "contract_type",
    "payment_method", "discount_applied", "price_increase_last_3m",
    "complaint_type", "survey_response",
]

# ── Entraînement ─────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Modèles ───────────────────────────────────────────────────────────────────
LOGISTIC_PARAMS = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__solver": ["lbfgs"],
    "classifier__max_iter": [1000],
}

RF_PARAMS = {
    "classifier__n_estimators": [200],
    "classifier__max_depth": [6, 10, None],
    "classifier__min_samples_leaf": [2, 5],
}

XGB_PARAMS = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [4, 6],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8],
    "classifier__colsample_bytree": [0.8],
}

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASH_PORT = 8501

# ── Seuils métier ─────────────────────────────────────────────────────────────
CHURN_THRESHOLD     = 0.4    # seuil optimisé (Recall > Precision pour minimiser FN)
HIGH_RISK_THRESHOLD = 0.6
