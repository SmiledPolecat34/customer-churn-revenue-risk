"""
Dashboard Streamlit — Analyse du Churn et Risque Revenus.

Sections :
    1. Vue d'ensemble — KPIs globaux du dataset
    2. Analyse exploratoire — distributions, corrélations
    3. Performance des modèles — métriques, courbes
    4. Prédiction individuelle — saisie manuelle d'un client
    5. Analyse SHAP — interprétabilité

Lancement :
    streamlit run dashboard/app.py
"""

import sys
import os
from pathlib import Path

# Ajout de la racine du projet au PATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from src.config import (
    RAW_CSV, METRICS_CSV, METRICS_DIR, FIGURES_DIR, SHAP_DIR,
    CHURN_THRESHOLD, HIGH_RISK_THRESHOLD, CHURN_MODEL_PATH,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.predict import predict_full, load_models

# ── Configuration Streamlit ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn & Revenue Risk — M1 Data Science",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style matplotlib professionnel ───────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
    "axes.edgecolor":    "#CCCCCC",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.8,
    "figure.dpi":        120,
})

PALETTE_CHURN    = "#C0392B"
PALETTE_NO_CHURN = "#1A5276"
PALETTE_BAR      = "#1A5276"
PALETTE_ACCENT   = "#2980B9"

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Typographie globale ─────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Header principal ────────────────────────────────── */
    .dash-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1A3A5C 100%);
        padding: 28px 32px 22px;
        border-radius: 12px;
        margin-bottom: 24px;
        border-left: 4px solid #2980B9;
    }
    .dash-header h1 {
        color: #FFFFFF !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        margin: 0 0 4px 0 !important;
        letter-spacing: -0.3px;
    }
    .dash-header p {
        color: #A8C4D8 !important;
        font-size: 0.82rem !important;
        margin: 0 !important;
    }
    .dash-header .badge {
        display: inline-block;
        background: rgba(41, 128, 185, 0.3);
        color: #7EC8E3;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid rgba(41,128,185,0.4);
        margin-top: 10px;
    }

    /* ── KPI Cards ───────────────────────────────────────── */
    .kpi-grid {
        display: flex; gap: 14px; flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .kpi-card {
        flex: 1; min-width: 140px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        border-top: 3px solid #1A5276;
    }
    .kpi-card .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1A202C;
        line-height: 1.1;
    }
    .kpi-card .kpi-sub {
        font-size: 0.75rem;
        color: #A0AEC0;
        margin-top: 4px;
    }
    .kpi-card.accent { border-top-color: #C0392B; }
    .kpi-card.green  { border-top-color: #27AE60; }
    .kpi-card.amber  { border-top-color: #D4AC0D; }

    /* ── Section headers ─────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1.5px solid #E2E8F0;
    }
    .section-header span {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2D3748;
        letter-spacing: -0.2px;
    }
    .section-dot {
        width: 8px; height: 8px;
        background: #1A5276;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Risk badges ─────────────────────────────────────── */
    .risk-high   { color: #C0392B; font-weight: 700; }
    .risk-medium { color: #D68910; font-weight: 700; }
    .risk-low    { color: #1E8449; font-weight: 700; }

    /* ── Result card ─────────────────────────────────────── */
    .result-card {
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1A5276;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* ── Images (mode normal uniquement, pas fullscreen) ── */
    [data-testid="stFullScreenFrame"]:not(.fullscreen) [data-testid="stImageContainer"] img {
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain;
        border-radius: 6px;
        display: block;
        margin: 0 auto;
    }
    /* plein écran : laisser l'image remplir le modal */
    [data-testid="stFullScreenFrame"].fullscreen [data-testid="stImageContainer"] img {
        width: 100% !important;
        max-width: 100% !important;
        max-height: 100% !important;
        object-fit: contain;
    }

    /* ── Streamlit overrides ─────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0D1B2A !important;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #A0AEC0 !important;
        font-size: 0.87rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2D3748 !important;
    }
    div[data-testid="metric-container"] {
        background: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px !important;
    }
    h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #1A202C !important; }
    h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #2D3748 !important; }
    h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #4A5568 !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stAlert { border-radius: 8px !important; }
    .stForm { border: 1px solid #E2E8F0 !important; border-radius: 10px !important; padding: 16px !important; }
</style>
""", unsafe_allow_html=True)


# ── Chargement des données (avec cache) ───────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(RAW_CSV)


@st.cache_data
def load_metrics():
    path = METRICS_CSV
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_churn_model():
    try:
        load_models()
        import joblib
        return joblib.load(CHURN_MODEL_PATH)
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="padding:18px 0 14px; text-align:center; border-bottom:1px solid #2D3748; margin-bottom:16px;">
    <div style="font-size:2rem; margin-bottom:6px;">📊</div>
    <div style="font-size:0.82rem; font-weight:700; color:#7EC8E3 !important; letter-spacing:0.04em; text-transform:uppercase;">
        Churn Dashboard
    </div>
    <div style="font-size:0.72rem; color:#718096 !important; margin-top:3px;">
        M1 Data Science · 2025-26
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Vue d'ensemble", "🔍 Analyse exploratoire",
     "📈 Performance modèles", "🎯 Prédiction client", "🧠 Interprétabilité SHAP"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.72rem; color:#4A5568 !important; line-height:1.7; padding: 0 4px;">
    <strong style="color:#718096 !important;">Auteurs</strong><br>
    VERSAYO Franklin<br>
    Danny Navarro Cordeau<br><br>
    <strong style="color:#718096 !important;">Encadrante</strong><br>
    Sarah Malaeb<br><br>
    <strong style="color:#718096 !important;">Stack</strong><br>
    Python · XGBoost · SHAP<br>
    FastAPI · Streamlit
</div>
""", unsafe_allow_html=True)

# ── Chargement ────────────────────────────────────────────────────────────────
df = load_data()
metrics_df = load_metrics()
model = load_churn_model()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Vue d'ensemble":
    st.markdown("""
    <div class="dash-header">
        <h1>Customer Churn &amp; Revenue Risk Dashboard</h1>
        <p>Prédiction du churn et estimation du risque de perte de revenus — dataset 10 000 clients SaaS</p>
        <span class="badge">Mastère Dev. Manager Full Stack · EFREI · 2025-26</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    churn_rate = df["churn"].mean()
    total_rev  = df["total_revenue"].sum()
    avg_rev    = df["total_revenue"].mean()
    n_churners = int(df["churn"].sum())

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Clients analysés</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-sub">27 features · split 80/20</div>
        </div>
        <div class="kpi-card accent">
            <div class="kpi-label">Taux de churn</div>
            <div class="kpi-value">{churn_rate:.1%}</div>
            <div class="kpi-sub">{n_churners} churners · déséquilibre 1:9</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Revenu total</div>
            <div class="kpi-value">{total_rev/1e6:.2f}M€</div>
            <div class="kpi-sub">ensemble du dataset</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Revenu moyen/client</div>
            <div class="kpi-value">{avg_rev:.0f}€</div>
            <div class="kpi-sub">médiane baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Distribution du churn</span></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["churn"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, autopct="%1.1f%%",
            colors=[PALETTE_NO_CHURN, PALETTE_CHURN],
            labels=["Non-Churn", "Churn"], startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Revenu total par segment</span></div>', unsafe_allow_html=True)
        if "customer_segment" in df.columns:
            seg_rev = df.groupby("customer_segment")["total_revenue"].sum().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = seg_rev.plot(kind="barh", ax=ax, color=PALETTE_BAR, width=0.6)
            ax.set_xlabel("Revenu total (€)")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
            for bar in ax.patches:
                ax.text(bar.get_width() + seg_rev.max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width()/1e6:.2f}M€",
                        va="center", fontsize=8, color="#444")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Tableau top clients à risque
    if model is not None:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#C0392B"></div><span>Top 10 clients à risque élevé</span></div>', unsafe_allow_html=True)
        try:
            feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            available   = [c for c in feature_cols if c in df.columns]
            X_all       = df[available]
            probas      = model.predict_proba(X_all)[:, 1]
            risk_df     = df.copy()
            risk_df["churn_proba"]      = probas
            risk_df["revenue_at_risk"]  = probas * df["total_revenue"]
            top_risk = (
                risk_df.sort_values("revenue_at_risk", ascending=False)
                .head(10)[["customer_id", "churn_proba", "total_revenue", "revenue_at_risk",
                            "customer_segment", "contract_type"]]
                .reset_index(drop=True)
            )
            top_risk["churn_proba"]     = top_risk["churn_proba"].map("{:.1%}".format)
            top_risk["revenue_at_risk"] = top_risk["revenue_at_risk"].map("{:.0f}€".format)
            st.dataframe(top_risk, use_container_width=True)
        except Exception as e:
            st.warning(f"Modèle non disponible : {e}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE EXPLORATOIRE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyse exploratoire":
    st.markdown("""
    <div class="dash-header">
        <h1>Analyse Exploratoire des Données</h1>
        <p>Distributions bivariées, taux de churn par modalité, corrélations entre features</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_available = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_available     = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Distribution — variable numérique</span></div>', unsafe_allow_html=True)
        num_feat = st.selectbox("Variable numérique", numeric_available, key="num_feat")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        df[df["churn"] == 0][num_feat].hist(ax=ax, bins=30, alpha=0.7,
            color=PALETTE_NO_CHURN, label="Non-Churn", edgecolor="white", linewidth=0.4)
        df[df["churn"] == 1][num_feat].hist(ax=ax, bins=30, alpha=0.75,
            color=PALETTE_CHURN, label="Churn", edgecolor="white", linewidth=0.4)
        ax.set_xlabel(num_feat)
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution · {num_feat}")
        ax.legend(frameon=True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Taux de churn par variable catégorielle</span></div>', unsafe_allow_html=True)
        cat_feat = st.selectbox("Variable catégorielle", cat_available, key="cat_feat")
        churn_by_cat = df.groupby(cat_feat)["churn"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        colors = [PALETTE_CHURN if v > df["churn"].mean() else PALETTE_ACCENT
                  for v in churn_by_cat.values]
        churn_by_cat.plot(kind="barh", ax=ax, color=colors, width=0.6)
        ax.set_xlabel("Taux de churn")
        ax.set_title(f"Churn rate · {cat_feat}")
        ax.axvline(df["churn"].mean(), color="#2C3E50", linestyle="--", lw=1.5, label=f"Moy. {df['churn'].mean():.1%}")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(frameon=True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('<div class="section-header"><div class="section-dot"></div><span>Matrice de corrélation — features numériques</span></div>', unsafe_allow_html=True)
    top_n = st.slider("Nombre de variables affichées", 5, len(numeric_available), 10)
    corr_matrix = df[numeric_available[:top_n]].corr()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, ax=ax, linewidths=0.5, linecolor="#FFFFFF",
                annot_kws={"size": 8}, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Corrélations entre features numériques")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE DES MODÈLES
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance modèles":
    st.markdown("""
    <div class="dash-header">
        <h1>Performance des Modèles de Classification</h1>
        <p>Comparaison Logistic Regression · Random Forest · XGBoost — métrique principale : PR-AUC</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics_df is not None:
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Tableau comparatif des métriques</span></div>', unsafe_allow_html=True)
        display_cols = ["model", "f1", "roc_auc", "pr_auc", "recall", "precision", "accuracy"]
        available_cols = [c for c in display_cols if c in metrics_df.columns]
        num_cols = [c for c in available_cols if c != "model"]
        styled = (
            metrics_df[available_cols].style
            .highlight_max(subset=num_cols, color="#D5E8D4", props="font-weight:bold;")
            .format({c: "{:.4f}" for c in num_cols})
            .set_properties(**{"font-size": "13px"})
        )
        st.dataframe(styled, use_container_width=True, height=130)
    else:
        st.info("Métriques non disponibles — lancez `python main.py` pour entraîner les modèles.")

    st.markdown('<div class="section-header"><div class="section-dot"></div><span>Courbes ROC</span></div>', unsafe_allow_html=True)
    roc_path = FIGURES_DIR / "roc_curves.png"
    if roc_path.exists():
        _c1, _c2, _c3 = st.columns([1, 1, 1])
        with _c2:
            st.image(str(roc_path), width=480)

    col1, col2 = st.columns(2)
    with col1:
        pr_path = FIGURES_DIR / "pr_curves.png"
        if pr_path.exists():
            st.markdown('<div class="section-header"><div class="section-dot"></div><span>Courbes Precision-Recall</span></div>', unsafe_allow_html=True)
            st.image(str(pr_path), width=400)
    with col2:
        comp_path = FIGURES_DIR / "models_comparison.png"
        if comp_path.exists():
            st.markdown('<div class="section-header"><div class="section-dot"></div><span>Comparaison des métriques</span></div>', unsafe_allow_html=True)
            st.image(str(comp_path), width=400)

    st.markdown('<div class="section-header"><div class="section-dot"></div><span>Matrices de confusion</span></div>', unsafe_allow_html=True)
    cm_files = list(FIGURES_DIR.glob("cm_*.png"))
    if cm_files:
        cols = st.columns(min(3, len(cm_files)))
        for i, cm_file in enumerate(cm_files):
            with cols[i % 3]:
                model_name = cm_file.stem.replace("cm_", "").replace("_", " ").title()
                st.caption(model_name)
                st.image(str(cm_file), width=300)
    else:
        st.info("Matrices de confusion non disponibles — lancez `python main.py`.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION INDIVIDUELLE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prédiction client":
    st.markdown("""
    <div class="dash-header">
        <h1>Prédiction individuelle — Profil client</h1>
        <p>Saisissez les caractéristiques du client pour estimer sa probabilité de churn et le revenue at risk</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("Modèle non chargé. Exécutez `python main.py` pour entraîner les modèles.")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("📋 Informations client")

        col1, col2, col3 = st.columns(3)
        with col1:
            age              = st.number_input("Âge", 18, 90, 35)
            tenure_months    = st.number_input("Ancienneté (mois)", 0, 120, 24)
            monthly_fee      = st.number_input("Frais mensuels (€)", 0.0, 500.0, 49.99)
            contract_type    = st.selectbox("Type de contrat", ["Monthly", "Annual", "Biannual"])
            payment_method   = st.selectbox("Moyen de paiement", ["Credit Card", "Bank Transfer", "PayPal", "Check"])
        with col2:
            monthly_logins     = st.number_input("Connexions/mois", 0, 100, 12)
            weekly_active_days = st.number_input("Jours actifs/semaine", 0, 7, 4)
            avg_session_time   = st.number_input("Durée session moy. (min)", 0.0, 300.0, 25.0)
            features_used      = st.number_input("Features utilisées", 0, 50, 8)
            last_login_days_ago = st.number_input("Dernière connexion (jours)", 0, 365, 3)
        with col3:
            support_tickets    = st.number_input("Tickets support", 0, 50, 2)
            csat_score         = st.slider("Score CSAT", 0.0, 10.0, 7.5)
            nps_score          = st.slider("NPS Score", -10, 10, 6)
            payment_failures   = st.number_input("Échecs de paiement", 0, 20, 0)
            customer_segment   = st.selectbox("Segment", ["SMB", "Enterprise", "Startup", "Consumer"])

        col4, col5 = st.columns(2)
        with col4:
            usage_growth_rate    = st.number_input("Taux de croissance usage", -1.0, 5.0, 0.05, step=0.01)
            avg_resolution_time  = st.number_input("Temps résolution moy. (h)", 0.0, 200.0, 24.0)
            escalations          = st.number_input("Escalations", 0, 20, 0)
            email_open_rate      = st.slider("Taux ouverture email", 0.0, 1.0, 0.35)
            marketing_click_rate = st.slider("Taux clic marketing", 0.0, 1.0, 0.1)
        with col5:
            referral_count        = st.number_input("Parrainages", 0, 30, 1)
            gender                = st.selectbox("Genre", ["Male", "Female", "Other"])
            signup_channel        = st.selectbox("Canal d'inscription", ["Online", "Referral", "Partner", "Direct"])
            discount_applied      = st.selectbox("Remise appliquée", ["No", "Yes"])
            price_increase_last_3m = st.selectbox("Hausse prix (3 mois)", ["No", "Yes"])
            complaint_type        = st.selectbox("Type plainte", ["None", "Billing", "Technical", "Service"])
            survey_response       = st.selectbox("Réponse enquête", ["Satisfied", "Neutral", "Dissatisfied", "Not Responded"])

        submitted = st.form_submit_button("🔮 Prédire", type="primary")

    if submitted:
        customer = {
            "age": age, "tenure_months": tenure_months,
            "monthly_logins": monthly_logins, "weekly_active_days": weekly_active_days,
            "avg_session_time": avg_session_time, "features_used": features_used,
            "usage_growth_rate": usage_growth_rate, "last_login_days_ago": last_login_days_ago,
            "monthly_fee": monthly_fee, "payment_failures": payment_failures,
            "support_tickets": support_tickets, "avg_resolution_time": avg_resolution_time,
            "csat_score": csat_score, "escalations": escalations,
            "email_open_rate": email_open_rate, "marketing_click_rate": marketing_click_rate,
            "nps_score": nps_score, "referral_count": referral_count,
            "gender": gender, "customer_segment": customer_segment,
            "signup_channel": signup_channel, "contract_type": contract_type,
            "payment_method": payment_method, "discount_applied": discount_applied,
            "price_increase_last_3m": price_increase_last_3m, "complaint_type": complaint_type,
            "survey_response": survey_response,
        }

        with st.spinner("Calcul en cours ..."):
            try:
                result = predict_full(customer)
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.stop()

        # Affichage résultats
        proba = result["churn_probability"]
        rar   = result["revenue_at_risk"]
        level = result["risk_level"]

        risk_color = "#C0392B" if level == "Élevé" else "#D68910" if level == "Moyen" else "#1E8449"
        risk_bg    = "#FDEDEC" if level == "Élevé" else "#FEF9E7" if level == "Moyen" else "#EAFAF1"

        st.markdown(f"""
        <div class="kpi-grid" style="margin-top:16px;">
            <div class="kpi-card {'accent' if level == 'Élevé' else ''}">
                <div class="kpi-label">Probabilité de churn</div>
                <div class="kpi-value">{proba:.1%}</div>
                <div class="kpi-sub">seuil décision : {CHURN_THRESHOLD:.0%}</div>
            </div>
            <div class="kpi-card amber">
                <div class="kpi-label">Revenue at Risk</div>
                <div class="kpi-value">{rar:.0f}€</div>
                <div class="kpi-sub">P(churn) × revenu total</div>
            </div>
            <div class="kpi-card" style="border-top-color:{risk_color}; background:{risk_bg};">
                <div class="kpi-label">Niveau de risque</div>
                <div class="kpi-value" style="color:{risk_color}; font-size:1.3rem;">{level}</div>
                <div class="kpi-sub">classification XGBoost</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Jauge de probabilité
        fig, ax = plt.subplots(figsize=(7, 1.0))
        bar_color = PALETTE_CHURN if proba >= HIGH_RISK_THRESHOLD else "#D68910" if proba >= CHURN_THRESHOLD else "#1E8449"
        ax.barh(0, proba, color=bar_color, height=0.45)
        ax.barh(0, 1 - proba, left=proba, color="#E8E8E8", height=0.45)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.axvline(CHURN_THRESHOLD, color="#2C3E50", lw=1.5, linestyle="--", label=f"Seuil {CHURN_THRESHOLD:.0%}")
        ax.axvline(HIGH_RISK_THRESHOLD, color="#922B21", lw=1.5, linestyle=":", label=f"Risque élevé {HIGH_RISK_THRESHOLD:.0%}")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title("Jauge de probabilité de churn")
        ax.legend(fontsize=8, loc="upper right", frameon=True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Recommandation
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.72rem; font-weight:700; color:#718096; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
                Recommandation
            </div>
            <div style="font-size:0.9rem; color:#2D3748; line-height:1.6;">
                {result['recommended_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Interprétabilité SHAP":
    st.markdown("""
    <div class="dash-header">
        <h1>Interprétabilité — Valeurs SHAP</h1>
        <p>SHapley Additive exPlanations · contribution de chaque feature à la prédiction du churn</p>
    </div>
    """, unsafe_allow_html=True)

    beeswarm = SHAP_DIR / "shap_summary_beeswarm.png"
    bar_plot  = SHAP_DIR / "shap_summary_bar.png"
    top_feat  = SHAP_DIR / "top_features_shap.csv"
    fi_plot   = FIGURES_DIR / "feature_importance_sklearn.png"

    if beeswarm.exists():
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Beeswarm — impact de chaque feature sur la prédiction</span></div>', unsafe_allow_html=True)
        _b1, _b2, _b3 = st.columns([1, 2, 1])
        with _b2:
            st.image(str(beeswarm), width=600)
    else:
        st.info("Graphique SHAP non disponible. Lancez `python main.py` (sans --skip-shap) pour le générer.")

    col1, col2 = st.columns(2)
    with col1:
        if bar_plot.exists():
            st.markdown('<div class="section-header"><div class="section-dot"></div><span>Importance absolue moyenne (SHAP)</span></div>', unsafe_allow_html=True)
            st.image(str(bar_plot), width=400)
    with col2:
        if fi_plot.exists():
            st.markdown('<div class="section-header"><div class="section-dot"></div><span>Feature importance — sklearn natif</span></div>', unsafe_allow_html=True)
            st.image(str(fi_plot), width=400)

    if top_feat.exists():
        st.markdown('<div class="section-header"><div class="section-dot"></div><span>Tableau — top features par valeur SHAP moyenne</span></div>', unsafe_allow_html=True)
        top_df = pd.read_csv(top_feat).head(20)
        st.dataframe(top_df, use_container_width=True, height=420)
