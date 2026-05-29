"""
Utilitaires partagés : logging, sauvegarde, chargement.
"""

import logging
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    MODELS_DIR, FIGURES_DIR, METRICS_DIR, SHAP_DIR,
    DATA_PROCESSED, REPORTS_DIR,
)


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré avec format lisible."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_dirs() -> None:
    """Crée tous les répertoires nécessaires s'ils n'existent pas."""
    for d in [MODELS_DIR, FIGURES_DIR, METRICS_DIR, SHAP_DIR, DATA_PROCESSED, REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_model(obj: Any, path: Path) -> None:
    """Sérialise un objet (modèle ou pipeline) avec joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    get_logger(__name__).info(f"Modèle sauvegardé → {path}")


def load_model(path: Path) -> Any:
    """Charge un objet sérialisé avec joblib."""
    logger = get_logger(__name__)
    if not Path(path).exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    logger.info(f"Modèle chargé ← {path}")
    return joblib.load(path)


def save_metrics(metrics: dict, path: Path) -> None:
    """Sauvegarde un dictionnaire de métriques en JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    get_logger(__name__).info(f"Métriques sauvegardées → {path}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame en Parquet ou CSV selon l'extension."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    get_logger(__name__).info(f"DataFrame sauvegardé ({len(df)} lignes) → {path}")
