"""
Module de prédiction — utilisé par l'API et le dashboard.

Fonctions :
    predict_churn(customer_data)     → probabilité de churn + label
    predict_revenue_risk(customer_data, churn_proba) → revenue at risk (€)
    predict_full(customer_data)      → résultat complet (churn + revenue)
    load_models()                    → charge les modèles une seule fois
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from src.config import (
    CHURN_MODEL_PATH, REVENUE_MODEL_PATH,
    CHURN_THRESHOLD, HIGH_RISK_THRESHOLD,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.utils import get_logger, load_model

logger = get_logger(__name__)

# ── Cache des modèles (chargés une seule fois au démarrage) ──────────────────
_models: dict = {}


def load_models() -> None:
    """Charge les modèles en mémoire. Appelé au démarrage de l'API."""
    global _models
    if not _models:
        logger.info("Chargement des modèles ...")
        _models["churn"]   = load_model(CHURN_MODEL_PATH)
        _models["revenue"] = load_model(REVENUE_MODEL_PATH)
        logger.info("Modèles chargés.")


def _to_dataframe(data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """Convertit un dict ou DataFrame en DataFrame avec les bonnes colonnes."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Type non supporté : {type(data)}")
    return df


def predict_churn(
    customer_data: Union[dict, pd.DataFrame],
    threshold: float = CHURN_THRESHOLD,
) -> dict:
    """
    Prédit la probabilité de churn et le label associé.

    Parameters
    ----------
    customer_data : dict ou DataFrame
        Données client (features brutes)
    threshold : float
        Seuil de classification (défaut : CHURN_THRESHOLD = 0.4)

    Returns
    -------
    dict avec :
        - churn_probability (float) : probabilité [0, 1]
        - churn_label (int)         : 0 ou 1
        - risk_level (str)          : "Faible", "Moyen", "Élevé"
        - threshold_used (float)
    """
    load_models()
    df = _to_dataframe(customer_data)

    model = _models["churn"]
    proba = model.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)

    results = []
    for p, l in zip(proba, label):
        if p >= HIGH_RISK_THRESHOLD:
            risk = "Élevé"
        elif p >= threshold:
            risk = "Moyen"
        else:
            risk = "Faible"
        results.append({
            "churn_probability": round(float(p), 4),
            "churn_label":       int(l),
            "risk_level":        risk,
            "threshold_used":    threshold,
        })

    return results[0] if len(results) == 1 else results


def predict_revenue_risk(
    customer_data: Union[dict, pd.DataFrame],
    churn_proba: float = None,
) -> dict:
    """
    Prédit le revenue at risk (€) pour un client.

    Si churn_proba n'est pas fourni, il est recalculé automatiquement.

    Returns
    -------
    dict avec :
        - revenue_at_risk (float) : montant en €
        - churn_probability (float)
        - interpretation (str)   : texte métier
    """
    load_models()
    df = _to_dataframe(customer_data)

    # Probabilité churn si non fournie
    if churn_proba is None:
        churn_result = predict_churn(df)
        if isinstance(churn_result, list):
            churn_proba = churn_result[0]["churn_probability"]
        else:
            churn_proba = churn_result["churn_probability"]

    model   = _models["revenue"]
    rar_raw = model.predict(df)       # revenu prédit pondéré par churn

    # Si le modèle prédit directement revenue_at_risk = churn_proba × revenue,
    # on le retourne directement ; sinon on applique la pondération
    revenue_at_risk = float(rar_raw[0]) if len(rar_raw) == 1 else float(np.mean(rar_raw))

    # Interprétation métier
    if revenue_at_risk > 500:
        interpretation = "⚠️ Risque élevé — action commerciale prioritaire recommandée"
    elif revenue_at_risk > 150:
        interpretation = "🔶 Risque modéré — suivi proactif conseillé"
    else:
        interpretation = "✅ Risque faible — client stable"

    return {
        "revenue_at_risk": round(revenue_at_risk, 2),
        "churn_probability": round(float(churn_proba), 4),
        "interpretation": interpretation,
    }


def predict_full(customer_data: Union[dict, pd.DataFrame]) -> dict:
    """
    Prédiction complète : churn + revenue at risk en un seul appel.

    Returns
    -------
    dict combinant churn et revenue risk, plus une recommandation métier.
    """
    churn_result   = predict_churn(customer_data)
    revenue_result = predict_revenue_risk(
        customer_data,
        churn_proba=churn_result["churn_probability"],
    )

    # Recommandation actionnable
    p = churn_result["churn_probability"]
    rar = revenue_result["revenue_at_risk"]

    if p >= HIGH_RISK_THRESHOLD and rar > 300:
        action = "PRIORITÉ CRITIQUE : contacter le client sous 24h, offrir un avantage de rétention (remise, upgrade)."
    elif p >= CHURN_THRESHOLD:
        action = "RISQUE MODÉRÉ : envoyer une campagne de rétention personnalisée dans la semaine."
    else:
        action = "Client stable — maintenir l'engagement par un programme de fidélité."

    return {
        **churn_result,
        **revenue_result,
        "recommended_action": action,
    }


def batch_predict(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prédictions en batch sur un DataFrame de clients.

    Returns
    -------
    DataFrame avec colonnes de prédiction ajoutées.
    """
    load_models()
    results = []
    for _, row in customers_df.iterrows():
        result = predict_full(row.to_dict())
        results.append(result)

    result_df = customers_df.copy().reset_index(drop=True)
    pred_df   = pd.DataFrame(results)
    return pd.concat([result_df, pred_df], axis=1)
