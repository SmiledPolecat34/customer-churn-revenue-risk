"""Tests unitaires — Module predict."""
import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CHURN_THRESHOLD, HIGH_RISK_THRESHOLD
from src.predict import predict_churn, predict_revenue_risk, predict_full

EXAMPLE_CUSTOMER = {
    "age": 35, "tenure_months": 24, "monthly_logins": 12,
    "weekly_active_days": 4, "avg_session_time": 25.5,
    "features_used": 8, "usage_growth_rate": 0.05,
    "last_login_days_ago": 3, "monthly_fee": 49.99,
    "payment_failures": 0, "support_tickets": 1,
    "avg_resolution_time": 12.0, "csat_score": 8.0,
    "escalations": 0, "email_open_rate": 0.4,
    "marketing_click_rate": 0.15, "nps_score": 7,
    "referral_count": 2,
    "gender": "Male", "customer_segment": "SMB",
    "signup_channel": "Online", "contract_type": "Annual",
    "payment_method": "Credit Card", "discount_applied": "Yes",
    "price_increase_last_3m": "No", "complaint_type": "None",
    "survey_response": "Satisfied",
}

def make_mocks(proba=0.3):
    mc = MagicMock()
    mc.predict_proba.return_value = np.array([[1-proba, proba]])
    mr = MagicMock()
    mr.predict.return_value = np.array([120.5])
    return {"churn": mc, "revenue": mr}

def test_churn_threshold_constants():
    assert 0.0 < CHURN_THRESHOLD < 1.0
    assert 0.0 < HIGH_RISK_THRESHOLD < 1.0
    assert HIGH_RISK_THRESHOLD > CHURN_THRESHOLD

def test_risk_level_faible():
    with patch("src.predict._models", make_mocks(proba=0.05)):
        result = predict_churn(EXAMPLE_CUSTOMER)
    assert result["risk_level"] == "Faible"
    assert result["churn_label"] == 0

def test_risk_level_moyen():
    proba = CHURN_THRESHOLD + 0.01
    with patch("src.predict._models", make_mocks(proba=proba)):
        result = predict_churn(EXAMPLE_CUSTOMER)
    assert result["risk_level"] == "Moyen"
    assert result["churn_label"] == 1

def test_risk_level_eleve():
    proba = HIGH_RISK_THRESHOLD + 0.01
    with patch("src.predict._models", make_mocks(proba=proba)):
        result = predict_churn(EXAMPLE_CUSTOMER)
    assert result["risk_level"] == "Élevé"
    assert result["churn_label"] == 1

def test_probability_in_range():
    with patch("src.predict._models", make_mocks(proba=0.3)):
        result = predict_churn(EXAMPLE_CUSTOMER)
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_revenue_risk_positive():
    with patch("src.predict._models", make_mocks(proba=0.3)):
        result = predict_revenue_risk(EXAMPLE_CUSTOMER, churn_proba=0.3)
    assert result["revenue_at_risk"] >= 0.0

def test_full_prediction_keys():
    with patch("src.predict._models", make_mocks(proba=0.45)):
        result = predict_full(EXAMPLE_CUSTOMER)
    for key in ["churn_probability","churn_label","risk_level","revenue_at_risk","interpretation","recommended_action"]:
        assert key in result, f"Cle manquante: {key}"

def test_recommended_action_not_empty():
    with patch("src.predict._models", make_mocks(proba=0.45)):
        result = predict_full(EXAMPLE_CUSTOMER)
    assert len(result["recommended_action"]) > 0
