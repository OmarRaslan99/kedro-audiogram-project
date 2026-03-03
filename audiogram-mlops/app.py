"""
API REST — audiogram-mlops
===========================
Expose les modèles tonal et vocal via Flask.

Routes :
    GET  /               → Statut de l'API et des modèles
    POST /predict         → Prédiction tonale (7 fréquences)
    POST /predict/vocal   → Prédiction vocale (scores + métadonnées)
    POST /train           → (Re)entraînement des pipelines Kedro
"""

import json
import logging
import os
import pickle
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_TONAL_PATH = BASE_DIR / "data" / "06_models" / "audiogram_model.pkl"
MODEL_VOCAL_PATH = BASE_DIR / "data" / "06_models" / "audiogram_vocal_model.pkl"
API_INPUTS_DIR = BASE_DIR / "data" / "07_api_inputs"

# Colonnes attendues (doit correspondre à features/nodes.py)
BEFORE_COLS = [
    "before_exam_125_Hz",
    "before_exam_250_Hz",
    "before_exam_500_Hz",
    "before_exam_1000_Hz",
    "before_exam_2000_Hz",
    "before_exam_4000_Hz",
    "before_exam_8000_Hz",
]
AFTER_COLS = [
    "after_exam_125_Hz",
    "after_exam_250_Hz",
    "after_exam_500_Hz",
    "after_exam_1000_Hz",
    "after_exam_2000_Hz",
    "after_exam_4000_Hz",
    "after_exam_8000_Hz",
]

# Intensités vocales attendues (0 à 100, pas de 5)
VOCAL_INTENSITIES = list(range(0, 105, 5))  # [0, 5, 10, …, 100]
VOCAL_SCORE_COLS = [f"score_{i}dB" for i in VOCAL_INTENSITIES]
VOCAL_SURDITE_TYPES = ["perception", "severe", "transmission"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_api_inputs_dir():
    """Crée le dossier de sauvegarde des requêtes utilisateur si nécessaire."""
    API_INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_request(route: str, data: dict):
    """Sauvegarde une requête utilisateur en JSONL (une ligne JSON par requête)."""
    _ensure_api_inputs_dir()
    filename = "tonal_requests.jsonl" if "vocal" not in route else "vocal_requests.jsonl"
    filepath = API_INPUTS_DIR / filename
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "api_user",
        "route": route,
        "data": data,
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Requête sauvegardée → %s", filepath)


def _load_model(path: Path):
    """Charge un modèle pickle depuis le disque."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Route GET /  — Accueil
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Page d'accueil : statut de l'API et disponibilité des modèles."""
    return jsonify({
        "project": "audiogram-mlops",
        "status": "ok",
        "models": {
            "tonal": MODEL_TONAL_PATH.exists(),
            "vocal": MODEL_VOCAL_PATH.exists(),
        },
        "routes": {
            "GET  /": "Statut de l'API",
            "POST /predict": "Prédiction tonale",
            "POST /predict/vocal": "Prédiction vocale",
            "POST /train": "(Re)entraînement des pipelines",
        },
    })


# ---------------------------------------------------------------------------
# Route POST /predict  — Prédiction tonale
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_tonal():
    """
    Reçoit 7 seuils auditifs avant intervention et retourne les 7 prédictions après.

    Body JSON attendu :
    {
        "before_exam_125_Hz": 30,
        "before_exam_250_Hz": 35,
        "before_exam_500_Hz": 40,
        "before_exam_1000_Hz": 45,
        "before_exam_2000_Hz": 50,
        "before_exam_4000_Hz": 55,
        "before_exam_8000_Hz": 60
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Bad Request", "message": "Le body doit être du JSON valide."}), 400

        # --- Validation des clés ---
        missing = [c for c in BEFORE_COLS if c not in data]
        if missing:
            return jsonify({
                "error": "Bad Request",
                "message": f"Clés manquantes : {missing}",
                "expected_keys": BEFORE_COLS,
            }), 400

        # --- Validation des valeurs ---
        values = {}
        for col in BEFORE_COLS:
            try:
                v = float(data[col])
            except (ValueError, TypeError):
                return jsonify({
                    "error": "Bad Request",
                    "message": f"La valeur de '{col}' doit être numérique. Reçu : {data[col]}",
                }), 400
            if not (0 <= v <= 120):
                return jsonify({
                    "error": "Bad Request",
                    "message": f"La valeur de '{col}' doit être entre 0 et 120 dB. Reçu : {v}",
                }), 400
            values[col] = v

        # --- Sauvegarde de la requête ---
        _save_request("/predict", data)

        # --- Chargement du modèle ---
        model = _load_model(MODEL_TONAL_PATH)
        if model is None:
            return jsonify({
                "error": "Not Found",
                "message": "Modèle tonal introuvable. Lancez d'abord POST /train ou make tonal.",
            }), 404

        # --- Prédiction ---
        df = pd.DataFrame([values], columns=BEFORE_COLS)
        prediction = model.predict(df)

        # Construire la réponse (7 fréquences prédites)
        result = {}
        pred_flat = prediction.flatten() if hasattr(prediction, "flatten") else prediction[0]
        for i, col in enumerate(AFTER_COLS):
            result[col] = round(float(pred_flat[i]), 2)

        logger.info("Prédiction tonale OK : %s", result)
        return jsonify({"prediction": result})

    except Exception as e:
        logger.error("Erreur /predict : %s", traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Route POST /predict/vocal  — Prédiction vocale
# ---------------------------------------------------------------------------
@app.route("/predict/vocal", methods=["POST"])
def predict_vocal():
    """
    Reçoit les données d'audiométrie vocale et retourne les prédictions SRT50 et gain.

    Body JSON attendu :
    {
        "is_aided": 1,
        "type_surdite": "perception",
        "scores": {
            "score_0dB": 0,
            "score_5dB": 2,
            "score_10dB": 5,
            ...
            "score_100dB": 85
        }
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Bad Request", "message": "Le body doit être du JSON valide."}), 400

        # --- Validation is_aided ---
        if "is_aided" not in data:
            return jsonify({"error": "Bad Request", "message": "Clé 'is_aided' manquante."}), 400
        try:
            is_aided = int(data["is_aided"])
        except (ValueError, TypeError):
            return jsonify({"error": "Bad Request", "message": "'is_aided' doit être 0 ou 1."}), 400
        if is_aided not in (0, 1):
            return jsonify({"error": "Bad Request", "message": "'is_aided' doit être 0 ou 1."}), 400

        # --- Validation type_surdite ---
        if "type_surdite" not in data:
            return jsonify({"error": "Bad Request", "message": "Clé 'type_surdite' manquante."}), 400
        type_surdite = data["type_surdite"]
        if type_surdite not in VOCAL_SURDITE_TYPES:
            return jsonify({
                "error": "Bad Request",
                "message": f"'type_surdite' doit être parmi {VOCAL_SURDITE_TYPES}. Reçu : '{type_surdite}'",
            }), 400

        # --- Validation scores ---
        if "scores" not in data or not isinstance(data["scores"], dict):
            return jsonify({"error": "Bad Request", "message": "Clé 'scores' manquante ou invalide (dict attendu)."}), 400

        scores = data["scores"]
        missing_scores = [s for s in VOCAL_SCORE_COLS if s not in scores]
        if missing_scores:
            return jsonify({
                "error": "Bad Request",
                "message": f"Scores manquants : {missing_scores}",
                "expected_scores": VOCAL_SCORE_COLS,
            }), 400

        score_values = {}
        for col in VOCAL_SCORE_COLS:
            try:
                v = float(scores[col])
            except (ValueError, TypeError):
                return jsonify({
                    "error": "Bad Request",
                    "message": f"Score '{col}' doit être numérique. Reçu : {scores[col]}",
                }), 400
            if not (0 <= v <= 100):
                return jsonify({
                    "error": "Bad Request",
                    "message": f"Score '{col}' doit être entre 0 et 100. Reçu : {v}",
                }), 400
            score_values[col] = v

        # --- Sauvegarde de la requête ---
        _save_request("/predict/vocal", data)

        # --- Chargement du modèle ---
        model = _load_model(MODEL_VOCAL_PATH)
        if model is None:
            return jsonify({
                "error": "Not Found",
                "message": "Modèle vocal introuvable. Lancez d'abord POST /train ou make vocal.",
            }), 404

        # --- Construction du DataFrame d'entrée ---
        row = {}
        row["is_aided"] = is_aided
        row.update(score_values)

        # One-hot encoding de type_surdite (même logique que features_vocal/nodes.py)
        for t in VOCAL_SURDITE_TYPES:
            row[f"type_surdite_{t}"] = 1 if type_surdite == t else 0

        df = pd.DataFrame([row])

        # S'assurer que l'ordre des colonnes correspond à celui du training
        # L'ordre attendu : is_aided, score_0dB..score_100dB, type_surdite_*
        expected_cols = (
            ["is_aided"]
            + VOCAL_SCORE_COLS
            + [f"type_surdite_{t}" for t in VOCAL_SURDITE_TYPES]
        )
        df = df[expected_cols]

        # --- Prédiction ---
        prediction = model.predict(df)
        pred = prediction[0] if len(prediction.shape) > 1 else prediction

        result = {
            "true_srt50": round(float(pred[0]), 2),
            "true_gain_db": round(float(pred[1]), 2),
        }

        logger.info("Prédiction vocale OK : %s", result)
        return jsonify({"prediction": result})

    except Exception as e:
        logger.error("Erreur /predict/vocal : %s", traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Route POST /train  — (Re)entraînement
# ---------------------------------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
    """
    Lance le (re)entraînement via les pipelines Kedro.

    Body JSON optionnel :
    {
        "pipeline": "tonal"   // "tonal", "vocal" ou "all" (défaut: "all")
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        pipeline_choice = data.get("pipeline", "all").lower()

        if pipeline_choice not in ("tonal", "vocal", "all"):
            return jsonify({
                "error": "Bad Request",
                "message": f"'pipeline' doit être 'tonal', 'vocal' ou 'all'. Reçu : '{pipeline_choice}'",
            }), 400

        # --- Import Kedro (fait ici pour ne pas ralentir le chargement de l'API) ---
        from kedro.framework.session import KedroSession
        from kedro.framework.startup import bootstrap_project

        # Bootstrapper le projet Kedro
        os.chdir(str(BASE_DIR))
        bootstrap_project(BASE_DIR)

        results = {}

        # --- Pipelines à exécuter ---
        pipelines_to_run = []
        if pipeline_choice in ("tonal", "all"):
            pipelines_to_run += ["ingestion", "features", "training", "evaluation"]
        if pipeline_choice in ("vocal", "all"):
            pipelines_to_run += ["ingestion_vocal", "features_vocal", "training_vocal", "evaluation_vocal"]

        for pipe_name in pipelines_to_run:
            logger.info("Exécution du pipeline : %s", pipe_name)
            with KedroSession.create(project_path=BASE_DIR) as session:
                output = session.run(pipeline_name=pipe_name)
                if output:
                    # Convertir les valeurs numpy en types Python natifs pour JSON
                    serializable = {}
                    for k, v in output.items():
                        if isinstance(v, dict):
                            serializable[k] = {
                                sk: sv if not isinstance(sv, (np.floating, np.integer)) else float(sv)
                                for sk, sv in v.items()
                            }
                        elif isinstance(v, (np.floating, np.integer)):
                            serializable[k] = float(v)
                        else:
                            serializable[k] = str(v)
                    results[pipe_name] = serializable

        logger.info("Entraînement terminé pour : %s", pipeline_choice)

        return jsonify({
            "status": "success",
            "pipeline": pipeline_choice,
            "message": f"Pipeline(s) '{pipeline_choice}' exécuté(s) avec succès.",
            "details": results,
        })

    except Exception as e:
        logger.error("Erreur /train : %s", traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Gestion d'erreurs globale
# ---------------------------------------------------------------------------
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": str(e)}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method Not Allowed", "message": str(e)}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _ensure_api_inputs_dir()
    logger.info("Démarrage de l'API audiogram-mlops sur http://0.0.0.0:5001")
    logger.info("Modèle tonal : %s (existe=%s)", MODEL_TONAL_PATH, MODEL_TONAL_PATH.exists())
    logger.info("Modèle vocal : %s (existe=%s)", MODEL_VOCAL_PATH, MODEL_VOCAL_PATH.exists())
    app.run(host="0.0.0.0", port=5001, debug=True)
