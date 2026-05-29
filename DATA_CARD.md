# DATA CARD — customer_churn.csv

> **Projet M1 Data Science — Mastère Dev. Manager Full Stack, 2025-26**  
> **Auteurs :** VERSAYO Franklin & Danny Navarro Cordeau

---

## Informations générales

| Champ | Valeur |
|-------|--------|
| Fichier | `data/raw/customer_churn.csv` |
| Format | CSV, UTF-8 |
| Taille | ~10 000 lignes |
| Colonnes | 30 |
| Variable cible | `churn` (0/1) |
| Cible secondaire | `total_revenue` (régression) |
| Identifiant client | `customer_id` |
| Taux de churn | ~10% (classe fortement déséquilibrée) |

---

## Description des colonnes

### Identifiant
- `customer_id` — identifiant unique client (exclu du feature set)

### Cibles
- `churn` — 1 si le client a quitté, 0 sinon
- `total_revenue` — revenu total généré par le client (€)

### Features numériques (18)

| Variable | Description |
|----------|-------------|
| `age` | Âge du client (années) |
| `tenure_months` | Ancienneté chez l'opérateur (mois) |
| `monthly_logins` | Nombre de connexions par mois |
| `weekly_active_days` | Jours actifs par semaine (0-7) |
| `avg_session_time` | Durée moyenne de session (minutes) |
| `features_used` | Nombre de fonctionnalités utilisées |
| `usage_growth_rate` | Taux de croissance de l'usage (%) |
| `last_login_days_ago` | Nombre de jours depuis la dernière connexion |
| `monthly_fee` | Frais mensuels facturés (€) |
| `payment_failures` | Nombre d'échecs de paiement |
| `support_tickets` | Nombre de tickets support ouverts |
| `avg_resolution_time` | Temps moyen de résolution d'un ticket (heures) |
| `csat_score` | Score de satisfaction client (0-10) |
| `escalations` | Nombre d'escalades support |
| `email_open_rate` | Taux d'ouverture des emails marketing (0-1) |
| `marketing_click_rate` | Taux de clic sur les emails marketing (0-1) |
| `nps_score` | Net Promoter Score (-10 à +10) |
| `referral_count` | Nombre de parrainages effectués |

### Features catégorielles (9)

| Variable | Modalités |
|----------|-----------|
| `gender` | Male, Female, Other |
| `customer_segment` | SMB, Enterprise, Startup, Consumer |
| `signup_channel` | Online, Referral, Partner, Direct |
| `contract_type` | Monthly, Annual, Biannual |
| `payment_method` | Credit Card, Bank Transfer, PayPal, Check |
| `discount_applied` | Yes, No |
| `price_increase_last_3m` | Yes, No |
| `complaint_type` | None, Billing, Technical, Service |
| `survey_response` | Satisfied, Neutral, Dissatisfied, Not Responded |

### Colonnes exclues du feature set
- `city` — haute cardinalité, peu informative
- `country` — haute cardinalité, peu informative

---

## Qualité des données

### Valeurs manquantes
Le dataset peut présenter des valeurs manquantes dans certaines colonnes numériques et catégorielles. Traitement appliqué :
- Colonnes numériques : **imputation par la médiane** (robuste aux outliers)
- Colonnes catégorielles : **imputation par "missing"** (modalité explicite)

### Outliers
Quelques variables présentent des valeurs extrêmes (ex. `payment_failures`, `support_tickets`, `avg_resolution_time`). Ces outliers sont **conservés** car ils représentent de vrais comportements à risque et constituent un signal prédictif pour le churn.

### Déséquilibre des classes
Le taux de churn est d'environ **10%**, soit un ratio 9:1 en faveur de la classe majoritaire (non-churn). Sans traitement, les modèles tendraient à ignorer la classe minoritaire.

**Stratégies de compensation appliquées :**
- SMOTE (Synthetic Minority Over-sampling Technique) sur le train set
- `class_weight="balanced"` pour Logistic Regression et Random Forest
- `scale_pos_weight=9` pour XGBoost
- Seuil de classification abaissé à 0.4 (favorise le Recall)

---

## Considérations éthiques

### Biais potentiels

**Genre (`gender`)** : Des différences de taux de churn selon le genre peuvent refléter des inégalités systémiques (tarification, service) plutôt que des comportements intrinsèques. Le modèle ne doit pas être utilisé pour discriminer des offres selon le genre.

**Âge (`age`)** : Une corrélation avec le churn ne justifie pas des offres différenciées contraires au droit à l'égalité de traitement.

**Segment client** : Des politiques de fidélisation différenciées selon le segment sont légitimes si basées sur la valeur économique, non sur des caractéristiques protégées.

### Recommandations d'usage éthique

1. Le modèle doit être utilisé pour **orienter des actions de rétention** (offres, support proactif), non pour exclure des clients de services.
2. Les décisions à fort impact sur le client (résiliation, majoration tarifaire) ne doivent **pas être entièrement automatisées** — un humain doit rester dans la boucle.
3. Les clients ciblés par les modèles doivent pouvoir **exercer un droit à l'explication** (RGPD Art. 22).
4. Le modèle doit être **réévalué régulièrement** (drift détection) car les comportements clients évoluent.

### Conformité RGPD
- Le `customer_id` est pseudonymisé et exclu du feature set
- Les données ne doivent pas être transmises à des tiers sans consentement
- Durée de conservation des données : à définir selon la politique de l'entreprise

### Limites du modèle
- Le dataset est **synthétique** — les performances réelles en production peuvent différer
- Le modèle ne capture pas les **événements externes** (crises économiques, offres concurrentes)
- La **causalité n'est pas établie** — les features corrélées au churn n'en sont pas forcément la cause
- Le modèle doit être retrained régulièrement pour éviter le **data drift**

---

## Reproductibilité

- `RANDOM_STATE = 42` fixé dans `src/config.py`
- Stratified split préservant le taux de churn (10%) dans train et test
- Pipeline sklearn anti data leakage (fit uniquement sur train)

---

*Dernière mise à jour : Mai 2026*
