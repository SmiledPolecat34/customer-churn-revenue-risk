# Customer Churn & Revenue Risk — Projet M1 Data Science

> **Mastère Dev. Manager Full Stack — 2025-26**  
> **Auteurs :** VERSAYO Franklin & Danny Navarro Cordeau  
> **Professeure :** Sarah Malaeb

---

## Contexte

Projet de Data Science de bout en bout visant à **anticiper le churn client** et estimer le **risque de perte de revenus** à partir d'un dataset de 10 000+ clients SaaS.

Deux modèles prédictifs sont développés :
- **Modèle de classification** (churn oui/non) → Logistic Regression, Random Forest, XGBoost
- **Modèle de régression** (revenue at risk en €) → Ridge, Random Forest, XGBoost

## Structure du projet

```
projet_final/rendu/
├── data/
│   ├── raw/customer_churn.csv       # Dataset brut
│   └── processed/                   # Splits train/test (parquet)
├── src/
│   ├── config.py                    # Configuration centrale (chemins, params)
│   ├── utils.py                     # Utilitaires partagés
│   ├── preprocessing.py             # Pipeline sklearn anti data leakage
│   ├── train_churn_models.py        # Entraînement classification (SMOTE)
│   ├── train_revenue_model.py       # Entraînement régression
│   ├── explainability.py            # SHAP — interprétabilité
│   └── predict.py                   # Fonctions de prédiction (API/dashboard)
├── api/
│   └── main.py                      # API REST FastAPI
├── dashboard/
│   └── app.py                       # Dashboard Streamlit interactif
├── notebooks/
│   └── eda.ipynb                    # Analyse exploratoire complète
├── models/                          # Modèles sérialisés (.joblib)
├── reports/
│   ├── figures/                     # Graphiques (ROC, CM, SHAP, EDA...)
│   ├── metrics/                     # CSV des métriques
│   └── shap/                        # Valeurs SHAP + figures
├── tests/
│   ├── test_preprocessing.py        # Tests unitaires preprocessing
│   └── test_predict.py              # Tests unitaires prédiction
├── main.py                          # Entrypoint pipeline complet
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
# 1. Cloner le repo / extraire l'archive
cd projet_final/rendu

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Exécution

### Pipeline complet (prétraitement + entraînement + SHAP)

```bash
python main.py
```

### Pipeline sans SHAP (plus rapide, ~2-3 min)

```bash
python main.py --skip-shap
```

### Tester la prédiction seule (après entraînement)

```bash
python main.py --only-predict
```

### Lancer l'API REST

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### Lancer le dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

Dashboard accessible sur : [http://localhost:8501](http://localhost:8501)

### Exécuter les tests unitaires

```bash
pytest tests/ -v
```

### Exécuter le notebook EDA

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Architecture ML

### Gestion du déséquilibre de classes (~10% churn)

- **SMOTE** intégré dans le pipeline `ImbPipeline` (appliqué sur train uniquement)
- `class_weight="balanced"` pour Logistic Regression et Random Forest
- `scale_pos_weight=9` pour XGBoost
- Seuil de classification abaissé à **0.4** (maximise le Recall)

### Anti data leakage

Le `ColumnTransformer` est fitted uniquement sur `X_train` (jamais sur le test set).  
Le SMOTE est encapsulé dans le pipeline → ne s'applique qu'au moment du `fit`.

### Métriques prioritaires

- Classification : **PR-AUC** (prioritaire sur données déséquilibrées) + F1, ROC-AUC, Recall
- Régression : **RMSE** + MAE, R²

---

## API REST — Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Statut de l'API |
| `POST` | `/predict/churn` | Probabilité de churn |
| `POST` | `/predict/revenue-risk` | Revenue at risk (€) |
| `POST` | `/predict/full` | Prédiction complète + recommandation |
| `POST` | `/predict/batch` | Batch jusqu'à 1000 clients |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict/full \
  -H "Content-Type: application/json" \
  -d '{
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
    "survey_response": "Dissatisfied"
  }'
```

### Réponse attendue

```json
{
  "churn_probability": 0.7832,
  "churn_label": 1,
  "risk_level": "Élevé",
  "threshold_used": 0.4,
  "revenue_at_risk": 457.23,
  "interpretation": "⚠️ Risque élevé — action commerciale prioritaire recommandée",
  "recommended_action": "PRIORITÉ CRITIQUE : contacter le client sous 24h, offrir un avantage de rétention."
}
```

---

## Reproductibilité

- `RANDOM_STATE = 42` — seed fixé dans `src/config.py`
- `TEST_SIZE = 0.2` — split stratifié préservant le taux de churn
- `CV_FOLDS = 5` — validation croisée stratifiée
- `requirements.txt` — versions de librairies fixées
- Scripts idempotents : réexécuter `python main.py` donne les mêmes résultats