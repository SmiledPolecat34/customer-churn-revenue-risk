"""
Tests unitaires — Module preprocessing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import build_preprocessor, build_features
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame minimal contenant toutes les features attendues."""
    n = 50
    rng = np.random.RandomState(42)

    numeric_data = {f: rng.randn(n) for f in NUMERIC_FEATURES}
    cat_data = {
        "gender": rng.choice(["Male", "Female"], n),
        "customer_segment": rng.choice(["SMB", "Enterprise", "Startup"], n),
        "signup_channel": rng.choice(["Online", "Referral"], n),
        "contract_type": rng.choice(["Monthly", "Annual"], n),
        "payment_method": rng.choice(["Credit Card", "Bank Transfer"], n),
        "discount_applied": rng.choice(["Yes", "No"], n),
        "price_increase_last_3m": rng.choice(["Yes", "No"], n),
        "complaint_type": rng.choice(["None", "Billing", "Technical"], n),
        "survey_response": rng.choice(["Satisfied", "Neutral", "Dissatisfied"], n),
    }
    # Introduire quelques NaN
    df = pd.DataFrame({**numeric_data, **cat_data})
    df.iloc[0, 0] = np.nan
    df.iloc[5, 1] = np.nan
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_preprocessor_no_nan(sample_df):
    """Le preprocessor doit imputer tous les NaN."""
    preprocessor = build_preprocessor()
    available_num = [c for c in NUMERIC_FEATURES if c in sample_df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in sample_df.columns]
    X = sample_df[available_num + available_cat]

    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    assert not np.isnan(X_transformed).any(), "Des NaN subsistent après transformation."


def test_build_preprocessor_shape(sample_df):
    """Le preprocessor doit retourner le bon nombre de colonnes."""
    preprocessor = build_preprocessor()
    available_num = [c for c in NUMERIC_FEATURES if c in sample_df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in sample_df.columns]
    X = sample_df[available_num + available_cat]

    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    assert X_transformed.shape[0] == len(sample_df), "Nombre de lignes incorrect."
    assert X_transformed.shape[1] > len(available_num), "OHE n'a pas augmenté le nombre de colonnes."


def test_no_data_leakage(sample_df):
    """Le fit du preprocessor doit être fait sur train uniquement."""
    preprocessor = build_preprocessor()
    available_num = [c for c in NUMERIC_FEATURES if c in sample_df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in sample_df.columns]
    X = sample_df[available_num + available_cat]

    train = X.iloc[:40]
    test  = X.iloc[40:]

    preprocessor.fit(train)  # fit sur train seulement
    X_train_t = preprocessor.transform(train)
    X_test_t  = preprocessor.transform(test)

    # La moyenne des colonnes numériques transformées (après StandardScaler)
    # doit être proche de 0 sur train, pas forcément sur test
    train_means = X_train_t[:, :len(available_num)].mean(axis=0)
    assert np.all(np.abs(train_means) < 1.0), "Scaling incorrect sur train."


def test_build_features_drops_city_country():
    """build_features doit supprimer 'city' et 'country' si présentes."""
    df = pd.DataFrame({
        "age": [25], "city": ["Paris"], "country": ["France"], "churn": [0]
    })
    df_out = build_features(df)
    assert "city" not in df_out.columns
    assert "country" not in df_out.columns
