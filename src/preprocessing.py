"""
Pipeline de prétraitement des données.
- Nettoyage, imputation, encodage, normalisation.
- Aucune fuite de données (fit uniquement sur train).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    RAW_CSV, PROCESSED_TRAIN, PROCESSED_TEST, PREPROCESSOR_PATH,
    TARGET_CHURN, TARGET_REVENUE, DROP_COLS, ID_COL,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_STATE,
)
from src.utils import get_logger, save_model, save_dataframe, ensure_dirs

logger = get_logger(__name__)


def load_raw_data(path=RAW_CSV) -> pd.DataFrame:
    """Charge le CSV brut et retourne un DataFrame propre."""
    df = pd.read_csv(path)
    logger.info(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def audit_data(df: pd.DataFrame) -> None:
    """Affiche un résumé de qualité des données."""
    logger.info("=== AUDIT DONNÉES ===")
    logger.info(f"Shape : {df.shape}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    mask = missing > 0
    if mask.any():
        logger.info(f"Valeurs manquantes :\n{pd.concat([missing[mask], missing_pct[mask]], axis=1, keys=['count', '%'])}")
    else:
        logger.info("Aucune valeur manquante.")
    logger.info(f"Distribution cible churn :\n{df[TARGET_CHURN].value_counts()}")
    logger.info(f"Taux de churn : {df[TARGET_CHURN].mean():.2%}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering léger avant preprocessing :
    - Supprime les colonnes non informatives.
    - Garde customer_id pour la traçabilité.
    """
    df = df.copy()

    # Suppression des colonnes de haute cardinalité non utiles
    cols_to_drop = [c for c in ["city", "country"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer sklearn :
    - Numérique : imputation médiane + StandardScaler
    - Catégoriel : imputation 'missing' + OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def run_preprocessing() -> tuple:
    """
    Pipeline complet :
    1. Charge et audite les données.
    2. Feature engineering.
    3. Split stratifié train/test.
    4. Fit du preprocessor sur train uniquement.
    5. Sauvegarde.

    Returns
    -------
    X_train, X_test, y_train_churn, y_test_churn,
    y_train_revenue, y_test_revenue, preprocessor
    """
    ensure_dirs()

    df = load_raw_data()
    audit_data(df)
    df = build_features(df)

    # Vérification des colonnes disponibles
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    logger.info(f"Features numériques : {len(available_num)}")
    logger.info(f"Features catégorielles : {len(available_cat)}")

    # Séparation X / y
    feature_cols = available_num + available_cat
    X = df[feature_cols]
    y_churn   = df[TARGET_CHURN].astype(int)
    y_revenue = df[TARGET_REVENUE].astype(float)

    # Split stratifié (préserve la proportion de churners)
    X_train, X_test, y_train_churn, y_test_churn, y_train_rev, y_test_rev = train_test_split(
        X, y_churn, y_revenue,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_churn,
    )
    logger.info(f"Train : {len(X_train)} | Test : {len(X_test)}")
    logger.info(f"Taux churn train : {y_train_churn.mean():.2%} | test : {y_test_churn.mean():.2%}")

    # Fit du preprocessor sur train uniquement → anti data leakage
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Sauvegarde preprocessor
    save_model(preprocessor, PREPROCESSOR_PATH)

    # Sauvegarde des splits (avec targets)
    train_df = X_train.copy()
    train_df[TARGET_CHURN]   = y_train_churn.values
    train_df[TARGET_REVENUE] = y_train_rev.values
    save_dataframe(train_df, PROCESSED_TRAIN)

    test_df = X_test.copy()
    test_df[TARGET_CHURN]   = y_test_churn.values
    test_df[TARGET_REVENUE] = y_test_rev.values
    save_dataframe(test_df, PROCESSED_TEST)

    return (X_train, X_test, y_train_churn, y_test_churn,
            y_train_rev, y_test_rev, preprocessor)


if __name__ == "__main__":
    run_preprocessing()
