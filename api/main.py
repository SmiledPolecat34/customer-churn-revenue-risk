"""
API REST — Prédiction de Churn et Revenue at Risk.

Endpoints :
    GET  /health                    → statut de l'API
    POST /predict/churn             → probabilité de churn
    POST /predict/revenue-risk      → revenue at risk (€)
    POST /predict/full              → prédiction complète + recommandation
    POST /predict/batch             → prédiction en batch (CSV JSON)

Lancement :
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.config import CHURN_THRESHOLD, HIGH_RISK_THRESHOLD
from src.predict import predict_churn, predict_revenue_risk, predict_full, load_models, batch_predict

# ── Configuration du logger ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn & Revenue Risk API",
    description=(
        "API de prédiction de churn et de risque revenus pour la rétention client. "
        "Projet M1 Data Science — VERSAYO Franklin & Danny Navarro Cordeau."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Toutes les features d'un client pour la prédiction."""
    # Numériques
    age:                   float = Field(..., ge=0, le=120, example=35)
    tenure_months:         float = Field(..., ge=0, example=24)
    monthly_logins:        float = Field(..., ge=0, example=12)
    weekly_active_days:    float = Field(..., ge=0, le=7, example=4)
    avg_session_time:      float = Field(..., ge=0, example=25.5)
    features_used:         float = Field(..., ge=0, example=8)
    usage_growth_rate:     float = Field(..., example=0.05)
    last_login_days_ago:   float = Field(..., ge=0, example=3)
    monthly_fee:           float = Field(..., ge=0, example=49.99)
    payment_failures:      float = Field(..., ge=0, example=0)
    support_tickets:       float = Field(..., ge=0, example=2)
    avg_resolution_time:   float = Field(..., ge=0, example=24.0)
    csat_score:            float = Field(..., ge=0, le=10, example=7.5)
    escalations:           float = Field(..., ge=0, example=0)
    email_open_rate:       float = Field(..., ge=0, le=1, example=0.35)
    marketing_click_rate:  float = Field(..., ge=0, le=1, example=0.1)
    nps_score:             float = Field(..., ge=-10, le=10, example=6)
    referral_count:        float = Field(..., ge=0, example=1)
    # Catégorielles
    gender:                str   = Field(..., example="Male")
    customer_segment:      str   = Field(..., example="SMB")
    signup_channel:        str   = Field(..., example="Online")
    contract_type:         str   = Field(..., example="Monthly")
    payment_method:        str   = Field(..., example="Credit Card")
    discount_applied:      str   = Field(..., example="No")
    price_increase_last_3m: str  = Field(..., example="No")
    complaint_type:        str   = Field(..., example="None")
    survey_response:       str   = Field(..., example="Satisfied")

    class Config:
        schema_extra = {
            "example": {
                "age": 35, "tenure_months": 24, "monthly_logins": 12,
                "weekly_active_days": 4, "avg_session_time": 25.5,
                "features_used": 8, "usage_growth_rate": 0.05,
                "last_login_days_ago": 3, "monthly_fee": 49.99,
                "payment_failures": 0, "support_tickets": 2,
                "avg_resolution_time": 24.0, "csat_score": 7.5,
                "escalations": 0, "email_open_rate": 0.35,
                "marketing_click_rate": 0.1, "nps_score": 6,
                "referral_count": 1,
                "gender": "Male", "customer_segment": "SMB",
                "signup_channel": "Online", "contract_type": "Monthly",
                "payment_method": "Credit Card", "discount_applied": "No",
                "price_increase_last_3m": "No", "complaint_type": "None",
                "survey_response": "Satisfied",
            }
        }


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_label:       int
    risk_level:        str
    threshold_used:    float


class RevenueRiskResponse(BaseModel):
    revenue_at_risk:   float
    churn_probability: float
    interpretation:    str


class FullPredictionResponse(BaseModel):
    churn_probability:  float
    churn_label:        int
    risk_level:         str
    threshold_used:     float
    revenue_at_risk:    float
    interpretation:     str
    recommended_action: str


class HealthResponse(BaseModel):
    status:    str
    timestamp: str
    version:   str


# ── Events ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage de l'API."""
    try:
        load_models()
        logger.info("✓ Modèles chargés avec succès.")
    except Exception as e:
        logger.error(f"✗ Erreur de chargement des modèles : {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Vérifie l'état de l'API."""
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version":   "1.0.0",
    }


@app.post("/predict/churn", response_model=ChurnResponse, tags=["Prediction"])
def predict_churn_endpoint(customer: CustomerFeatures):
    """
    Prédit la probabilité de churn d'un client.

    - **churn_probability** : score entre 0 et 1
    - **risk_level** : Faible / Moyen / Élevé
    """
    try:
        data = customer.model_dump()
        result = predict_churn(data)
        return result
    except Exception as e:
        logger.error(f"Erreur predict_churn : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/revenue-risk", response_model=RevenueRiskResponse, tags=["Prediction"])
def predict_revenue_endpoint(customer: CustomerFeatures):
    """
    Prédit le revenue at risk (€) = P(churn) × revenu_client.
    """
    try:
        data = customer.model_dump()
        result = predict_revenue_risk(data)
        return result
    except Exception as e:
        logger.error(f"Erreur predict_revenue : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full", response_model=FullPredictionResponse, tags=["Prediction"])
def predict_full_endpoint(customer: CustomerFeatures):
    """
    Prédiction complète : churn + revenue at risk + recommandation métier.
    """
    try:
        data = customer.model_dump()
        result = predict_full(data)
        return result
    except Exception as e:
        logger.error(f"Erreur predict_full : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch_endpoint(customers: list[CustomerFeatures]):
    """
    Prédiction en batch sur une liste de clients (max 1000).
    Retourne une liste de résultats complets.
    """
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch limité à 1000 clients.")
    try:
        df = pd.DataFrame([c.dict() for c in customers])
        result_df = batch_predict(df)
        pred_cols = [
            "churn_probability", "churn_label", "risk_level",
            "revenue_at_risk", "interpretation", "recommended_action",
        ]
        available = [c for c in pred_cols if c in result_df.columns]
        return result_df[available].to_dict(orient="records")
    except Exception as e:
        logger.error(f"Erreur predict_batch : {e}")
        raise HTTPException(status_code=500, detail=str(e))
