# REPORT — Projet Data Science M1 : Prédiction du Churn Client

> **Mastère Dev. Manager Full Stack — 2025-26**  
> **Auteurs :** VERSAYO Franklin & Danny Navarro Cordeau  
> **Professeure :** Sarah Malaeb  
> **Date :** Mai 2026

---

## 1. Introduction et problématique métier

Dans un contexte SaaS compétitif, **anticiper le départ d'un client (churn)** est un enjeu critique : acquérir un nouveau client coûte 5 à 7 fois plus cher que fidéliser un client existant. Ce projet vise à construire un système de prédiction end-to-end permettant à une équipe Customer Success de :

1. Identifier les clients à risque de churn avant leur départ effectif
2. Estimer le **revenue at risk** associé (€) pour prioriser les actions
3. Comprendre les leviers d'action grâce à l'interprétabilité des modèles

Le dataset contient **~10 000 clients** avec 27 features (18 numériques, 9 catégorielles) et un taux de churn de **~10%** — problème de classification binaire avec **déséquilibre fort**.

---

## 2. Pipeline Data Science

### 2.1 Audit des données

L'audit initial révèle :
- Dataset de bonne qualité avec un faible taux de valeurs manquantes (<5% par colonne)
- Déséquilibre fort : ~10% de churners (ratio 9:1)
- Outliers présents sur `payment_failures`, `support_tickets`, `avg_resolution_time` — conservés car porteurs d'information
- Variables `city` et `country` exclues (haute cardinalité, peu informatives)

### 2.2 Prétraitement

Le pipeline sklearn anti-data-leakage est implémenté via `ColumnTransformer` :

- **Variables numériques** : imputation par la médiane + `StandardScaler`
- **Variables catégorielles** : imputation par "missing" + `OneHotEncoder`
- **Split** : train/test stratifié (80/20), préservant le taux de churn
- Le preprocessor est fitté **uniquement sur le train set** pour éviter toute fuite de données

### 2.3 Feature Engineering

Au-delà des features brutes, la **cible de régression** est construite comme :

```
revenue_at_risk = P(churn) × total_revenue
```

Ce score métier combine probabilité de départ et valeur économique du client.

---

## 3. Modélisation — Classification (Churn)

### 3.1 Choix des modèles

Trois modèles ont été comparés selon la complexité croissante :
- **Logistic Regression** (baseline interprétable, linéaire)
- **Random Forest** (ensemble non-linéaire, robuste au bruit)
- **XGBoost** (gradient boosting, état de l'art sur données tabulaires)

Chaque modèle est encapsulé dans un `ImbPipeline` incluant **SMOTE** (appliqué uniquement sur train).

### 3.2 Gestion du déséquilibre

Trois mécanismes combinés :
1. **SMOTE** : sur-échantillonnage synthétique de la classe minoritaire (k=3 voisins)
2. `class_weight="balanced"` pour LR et RF
3. `scale_pos_weight=9` pour XGBoost
4. **Seuil abaissé à 0.4** (vs 0.5 standard) pour maximiser le Recall

Justification : dans ce contexte, un **faux négatif** (manquer un churner) est plus coûteux qu'un faux positif (action de rétention inutile).

### 3.3 Résultats et métriques

| Modèle | F1 | ROC-AUC | PR-AUC | Recall | Precision |
|--------|----|---------|--------|--------|-----------|
| Logistic Regression | 0.2555 | 0.7161 | 0.2357 | 0.7647 | 0.1534 |
| Random Forest | 0.2822 | 0.7704 | 0.2317 | 0.3333 | 0.2446 |
| **XGBoost** | **0.3731** | **0.7850** | **0.2621** | **0.7206** | **0.2517** |

*Seuil de classification : 0.4 — Dataset : 10 000 clients, 10.21% churn, split 80/20 stratifié.*

**Métrique de sélection : PR-AUC** (prioritaire sur ROC-AUC pour données déséquilibrées — l'AUC ROC est optimiste quand la classe négative est majoritaire).

### 3.4 Validation croisée

Cross-validation stratifiée à 5 folds sur le train set. Les scores CV F1 confirment la robustesse des modèles (faible écart-type inter-folds).

---

## 4. Modélisation — Régression (Revenue at Risk)

### 4.1 Construction de la cible

La cible `revenue_at_risk = P(churn) × total_revenue` combine :
- La **probabilité de churn** issue du meilleur modèle de classification
- Le **revenu total** du client

### 4.2 Modèles comparés

| Modèle | MAE (€) | RMSE (€) | R² |
|--------|---------|----------|-----|
| Ridge | 206.05 | 341.40 | 0.441 |
| Random Forest | 95.02 | 183.75 | 0.838 |
| **XGBoost** | **88.02** | **170.72** | **0.860** |

Revenue at risk moyen sur le test set : **279€** (max : 4 471€). Le meilleur modèle est XGBoost (R²=0.860).

---

## 5. Interprétabilité — SHAP

Les valeurs SHAP (SHapley Additive exPlanations) permettent d'expliquer les prédictions :

**Top features pour le churn (interprétation attendue) :**
- `last_login_days_ago` ↑ → churn plus probable (client inactif)
- `tenure_months` ↓ → churn plus probable (client récent)
- `payment_failures` ↑ → churn plus probable (friction financière)
- `csat_score` ↓ → churn plus probable (insatisfaction)
- `contract_type = Monthly` → churn plus probable (engagement faible)
- `nps_score` ↓ → churn plus probable (non-promoteur)

Les graphiques SHAP beeswarm et bar plot sont sauvegardés dans `reports/shap/`.

---

## 6. Exposition — POC

### API REST (FastAPI)
Endpoint `/predict/full` retourne en JSON :
- Probabilité de churn
- Niveau de risque (Faible / Moyen / Élevé)
- Revenue at risk (€)
- Recommandation actionnable (texte)

### Dashboard Streamlit
5 sections interactives : vue d'ensemble KPIs, analyse exploratoire, performance modèles, prédiction individuelle, interprétabilité SHAP.

---

## 7. Analyse d'erreurs et limites

### Faux négatifs (churners non détectés)
Avec un seuil de 0.4, le Recall est optimisé mais des churners restent non détectés. L'ajustement du seuil est un arbitrage métier entre coût des faux positifs (actions de rétention inutiles) et faux négatifs (churners manqués).

### Limites du modèle
1. **Dataset synthétique** : les performances en production réelle peuvent différer
2. **Pas de dimension temporelle** : le dataset est une photo statique ; un modèle séquentiel (LSTM) pourrait capturer des tendances
3. **Causalité vs corrélation** : les features corrélées au churn n'en sont pas nécessairement la cause
4. **Data drift** : les comportements clients évoluent ; le modèle doit être retrained périodiquement
5. **Interprétabilité locale** : SHAP global ne garantit pas l'explication correcte de chaque prédiction individuelle

### Biais identifiés
- Si le genre ou le segment sont corrélés au churn, le modèle peut amplifier des inégalités existantes
- Le SMOTE peut créer des exemples synthétiques non réalistes dans les espaces de features catégorielles

---

## 8. Recommandations actionnables

1. **Déployer le score de churn en temps réel** dans le CRM pour alerter les Customer Success Managers
2. **Prioriser par revenue at risk** : intervenir en premier sur les clients à fort revenu et haute probabilité
3. **Campagnes segmentées** : actions différenciées selon le niveau de risque (Faible/Moyen/Élevé)
4. **Amélioration CSAT et NPS** : investir dans la qualité du support (tickets, temps de résolution)
5. **Réduction des payment_failures** : proposer des solutions de paiement alternatives aux clients à risque
6. **Incentiver les contrats longue durée** (Annual vs Monthly) : signal fort de rétention
7. **Programme de re-engagement** pour les clients inactifs (`last_login_days_ago` élevé)

---

## 9. Conclusion

Ce projet démontre un pipeline Data Science complet et reproductible : de l'audit des données brutes à l'exposition via API et dashboard. Le modèle XGBoost avec SMOTE et seuil optimisé offre les meilleures performances sur ce problème déséquilibré, mesuré par la PR-AUC. L'interprétabilité SHAP permet de traduire les prédictions en leviers d'action concrets pour les équipes métier.

---

*Figures disponibles : `reports/figures/` | Métriques : `reports/metrics/` | Code source : `src/`*
