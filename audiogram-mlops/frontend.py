"""
Frontend Streamlit — audiogram-mlops
=====================================
Interface utilisateur pour les modèles d'audiométrie tonale et vocale.

Lancer :
    streamlit run frontend.py
    (l'API Flask doit tourner sur http://127.0.0.1:5001)
"""

import json
import os

import numpy as np
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:5001")

FREQ_LABELS = ["125", "250", "500", "1000", "2000", "4000", "8000"]
BEFORE_COLS = [f"before_exam_{f}_Hz" for f in FREQ_LABELS]
AFTER_COLS = [f"after_exam_{f}_Hz" for f in FREQ_LABELS]
VOCAL_INTENSITIES = list(range(0, 105, 5))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Audiogram MLOps",
    page_icon="🔊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    /* Status badge */
    .status-ok { color: #00c853; font-weight: 600; }
    .status-ko { color: #ff1744; font-weight: 600; }
    /* Divider */
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def api_status():
    """Vérifie si l'API Flask est accessible."""
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, None


def call_api(endpoint: str, payload: dict):
    """Appelle un endpoint POST de l'API."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 0, {"error": "Connexion refusée", "message": "L'API Flask ne répond pas. Lancez-la avec `make api`."}
    except Exception as e:
        return 0, {"error": "Erreur", "message": str(e)}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔊 Audiogram MLOps")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🎵 Prédiction Tonale", "🗣️ Prédiction Vocale", "🔧 Entraînement"],
        index=0,
    )

    st.markdown("---")

    # API status indicator
    ok, info = api_status()
    if ok:
        st.markdown("**API** : <span class='status-ok'>● En ligne</span>", unsafe_allow_html=True)
        if info:
            models = info.get("models", {})
            tonal_ok = models.get("tonal", False)
            vocal_ok = models.get("vocal", False)
            tonal_span = "<span class='status-ok'>● Prêt</span>" if tonal_ok else "<span class='status-ko'>● Absent</span>"
            vocal_span = "<span class='status-ok'>● Prêt</span>" if vocal_ok else "<span class='status-ko'>● Absent</span>"
            st.markdown(
                f"**Modèle tonal** : {tonal_span}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Modèle vocal** : {vocal_span}",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("**API** : <span class='status-ko'>● Hors ligne</span>", unsafe_allow_html=True)
        st.caption("Lancez l'API : `make api`")

    st.markdown("---")
    st.caption("Kedro 1.2 · MLflow · Flask · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE : Accueil
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown("<div class='main-header'><h1>🔊 Audiogram MLOps</h1></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:1.1rem; color:gray;'>"
        "Plateforme d'audiométrie prédictive — Tonale & Vocale"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎵 Audiométrie Tonale</h3>
            <div class='value'>7 fréquences</div>
            <p style='margin:0; font-size:0.8rem;'>125 Hz → 8000 Hz</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <h3>🗣️ Audiométrie Vocale</h3>
            <div class='value'>SRT50 + Gain</div>
            <p style='margin:0; font-size:0.8rem;'>21 intensités · 3 profils</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <h3>🔧 MLOps Pipeline</h3>
            <div class='value'>8 pipelines</div>
            <p style='margin:0; font-size:0.8rem;'>Kedro · MLflow · Optuna</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("### Comment ça fonctionne")
    st.markdown("""
    1. **Prédiction Tonale** — Saisissez les seuils auditifs (avant intervention) sur 7 fréquences,
       et le modèle prédit les seuils post-intervention.
    2. **Prédiction Vocale** — Saisissez la courbe vocale d'un patient (sans appareil),
       et le modèle prédit le SRT50 et le gain potentiel avec appareil auditif.
    3. **Entraînement** — Relancez l'entraînement des modèles directement depuis l'interface.
    """)

    st.info("💡 Assurez-vous que l'API Flask tourne (`make api`) avant d'utiliser les fonctions de prédiction.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE : Prédiction Tonale
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎵 Prédiction Tonale":
    st.markdown("## 🎵 Prédiction Tonale")
    st.markdown("Saisissez les seuils auditifs **avant intervention** (en dB HL) pour chaque fréquence.")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # --- Input sliders ---
    st.markdown("### Seuils avant intervention")
    cols = st.columns(len(FREQ_LABELS))
    values = {}
    # Default values simulating moderate hearing loss
    defaults = [25, 30, 35, 40, 50, 55, 60]

    for i, (col, freq, default) in enumerate(zip(cols, FREQ_LABELS, defaults)):
        with col:
            values[BEFORE_COLS[i]] = st.slider(
                f"{freq} Hz",
                min_value=0,
                max_value=120,
                value=default,
                step=5,
                key=f"tonal_{freq}",
            )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # --- Submit ---
    if st.button("🔮 Prédire les seuils après intervention", type="primary", use_container_width=True):
        with st.spinner("Appel de l'API en cours..."):
            status, result = call_api("/predict", values)

        if status == 200 and "prediction" in result:
            pred = result["prediction"]
            st.success("Prédiction réussie !")

            # --- Results ---
            st.markdown("### Résultats prédits (après intervention)")
            result_cols = st.columns(len(FREQ_LABELS))
            for i, (col, freq) in enumerate(zip(result_cols, FREQ_LABELS)):
                with col:
                    before_val = values[BEFORE_COLS[i]]
                    after_val = pred[AFTER_COLS[i]]
                    delta = round(after_val - before_val, 1)
                    st.metric(
                        label=f"{freq} Hz",
                        value=f"{after_val:.1f} dB",
                        delta=f"{delta:+.1f} dB",
                        delta_color="inverse",  # Lower is better for hearing
                    )

            # --- Chart ---
            st.markdown("### Audiogramme comparatif")
            import pandas as pd

            freqs_num = [int(f) for f in FREQ_LABELS]
            chart_data = pd.DataFrame({
                "Fréquence (Hz)": freqs_num * 2,
                "Seuil (dB HL)": (
                    [values[c] for c in BEFORE_COLS]
                    + [pred[c] for c in AFTER_COLS]
                ),
                "Condition": ["Avant intervention"] * 7 + ["Après intervention (prédit)"] * 7,
            })
            st.line_chart(
                chart_data,
                x="Fréquence (Hz)",
                y="Seuil (dB HL)",
                color="Condition",
                use_container_width=True,
            )
            st.caption("⬇️ Plus le seuil est bas, meilleure est l'audition.")

        else:
            st.error(f"Erreur ({status}) : {result.get('message', json.dumps(result))}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE : Prédiction Vocale
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🗣️ Prédiction Vocale":
    st.markdown("## 🗣️ Prédiction Vocale")
    st.markdown(
        "Saisissez la courbe vocale du patient **sans appareil** (`is_aided=0`). "
        "Le modèle prédit le **SRT50** et le **gain potentiel** avec appareil."
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # --- Metadata ---
    st.markdown("### Profil patient")
    meta_col1, meta_col2 = st.columns(2)

    with meta_col1:
        type_surdite = st.selectbox(
            "Type de surdité",
            ["perception", "transmission", "severe"],
            index=0,
            help="Profil audiométrique du patient",
        )

    with meta_col2:
        is_aided = st.selectbox(
            "Condition du test",
            [0, 1],
            index=0,
            format_func=lambda x: "Sans appareil (unaided)" if x == 0 else "Avec appareil (aided)",
            help="0 = test sans appareil (cas principal), 1 = test avec appareil",
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # --- Scores ---
    st.markdown("### Scores de reconnaissance vocale (%)")
    st.caption("Ajustez les scores pour chaque intensité (0–100 dB, par pas de 5 dB).")

    # Generate a default sigmoid curve for perception unaided
    def default_sigmoid(intensities, srt50=40, slope=0.15, max_score=88):
        scores = []
        for db in intensities:
            s = max_score / (1 + np.exp(-slope * (db - srt50)))
            if db > srt50 + 35:
                s -= 0.15 * (db - srt50 - 35)
            scores.append(max(0, min(100, round(s))))
        return scores

    defaults = default_sigmoid(VOCAL_INTENSITIES)

    # Display in 3 rows of 7 columns
    scores = {}
    rows = [VOCAL_INTENSITIES[i:i + 7] for i in range(0, len(VOCAL_INTENSITIES), 7)]
    default_rows = [defaults[i:i + 7] for i in range(0, len(defaults), 7)]

    for row_intensities, row_defaults in zip(rows, default_rows):
        cols = st.columns(len(row_intensities))
        for col, intensity, default_val in zip(cols, row_intensities, row_defaults):
            with col:
                scores[f"score_{intensity}dB"] = st.number_input(
                    f"{intensity} dB",
                    min_value=0,
                    max_value=100,
                    value=default_val,
                    step=1,
                    key=f"vocal_{intensity}",
                )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # --- Submit ---
    if st.button("🔮 Prédire SRT50 et gain", type="primary", use_container_width=True):
        payload = {
            "is_aided": is_aided,
            "type_surdite": type_surdite,
            "scores": scores,
        }

        with st.spinner("Appel de l'API en cours..."):
            status, result = call_api("/predict/vocal", payload)

        if status == 200 and "prediction" in result:
            pred = result["prediction"]
            st.success("Prédiction réussie !")

            # --- Results ---
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>SRT50 prédit</h3>
                    <div class='value'>{pred['true_srt50']:.1f} dB</div>
                    <p style='margin:0; font-size:0.8rem;'>Seuil de reconnaissance à 50%</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                    <h3>Gain prédit avec appareil</h3>
                    <div class='value'>{pred['true_gain_db']:.1f} dB</div>
                    <p style='margin:0; font-size:0.8rem;'>Amélioration attendue du SRT50</p>
                </div>
                """, unsafe_allow_html=True)

            # --- Vocal curve chart ---
            st.markdown("### Courbe vocale saisie")
            import pandas as pd

            chart_data = pd.DataFrame({
                "Intensité (dB)": VOCAL_INTENSITIES,
                "Score de reconnaissance (%)": [scores[f"score_{i}dB"] for i in VOCAL_INTENSITIES],
            })
            st.line_chart(chart_data, x="Intensité (dB)", y="Score de reconnaissance (%)", use_container_width=True)

            # --- Interpretation ---
            st.markdown("### Interprétation")
            srt50 = pred["true_srt50"]
            gain = pred["true_gain_db"]
            aided_srt50 = srt50 - gain

            if gain > 25:
                severity = "🟢 Excellent gain attendu"
            elif gain > 15:
                severity = "🟡 Gain modéré attendu"
            else:
                severity = "🔴 Gain limité attendu"

            st.markdown(f"""
            | Paramètre | Valeur |
            |---|---|
            | **SRT50 actuel** | {srt50:.1f} dB |
            | **Gain prédit** | {gain:.1f} dB |
            | **SRT50 aidé estimé** | {aided_srt50:.1f} dB |
            | **Pronostic** | {severity} |
            """)

        else:
            st.error(f"Erreur ({status}) : {result.get('message', json.dumps(result))}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE : Entraînement
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔧 Entraînement":
    st.markdown("## 🔧 (Re)entraînement des modèles")
    st.markdown("Lancez les pipelines Kedro directement depuis l'interface.")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    pipeline_choice = st.radio(
        "Pipeline à exécuter",
        ["all", "tonal", "vocal"],
        index=0,
        format_func=lambda x: {
            "all": "🔄 Tous les pipelines (tonal + vocal)",
            "tonal": "🎵 Pipeline tonal uniquement",
            "vocal": "🗣️ Pipeline vocal uniquement",
        }[x],
        horizontal=True,
    )

    st.warning("⚠️ L'entraînement peut prendre plusieurs minutes. L'API doit rester accessible.")

    if st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True):
        with st.spinner(f"Entraînement du pipeline **{pipeline_choice}** en cours... Veuillez patienter."):
            status, result = call_api("/train", {"pipeline": pipeline_choice})

        if status == 200 and result.get("status") == "success":
            st.success(f"✅ {result.get('message', 'Entraînement terminé !')}")
            st.balloons()

            # Show details
            details = result.get("details", {})
            if details:
                st.markdown("### Détails des pipelines exécutés")
                for pipe_name, pipe_output in details.items():
                    with st.expander(f"📦 {pipe_name}"):
                        st.json(pipe_output)
        else:
            st.error(f"Erreur ({status}) : {result.get('message', json.dumps(result))}")
