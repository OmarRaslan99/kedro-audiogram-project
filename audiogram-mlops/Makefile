# ============================================================================
#  Makefile — audiogram-mlops (Kedro + MLflow)
# ============================================================================
#  Usage :
#    make clean   → supprime tous les artefacts générés (garde les raw data)
#    make run     → lance tous les pipelines + kedro-viz + mlflow ui
#    make run-tonal   → lance uniquement les 4 pipelines tonaux
#    make run-vocal   → lance uniquement les 4 pipelines vocaux
#    make stop    → arrête les serveurs kedro-viz et mlflow ui
# ============================================================================

SHELL := /bin/bash

# ── Fichiers / dossiers générés à nettoyer ──────────────────────────────────

# MLflow
CLEAN_DIRS  := mlruns outputs .viz
CLEAN_FILES := mlflow.db info.log

# Données intermédiaires et modèles (pas les raw)
CLEAN_DATA_DIRS := data/02_intermediate data/05_model_input data/06_models

# Python cache
PYCACHE := $(shell find . -type d -name '__pycache__' 2>/dev/null)
EGG_INFO := src/audiogram_mlops.egg-info

# ── Cibles ──────────────────────────────────────────────────────────────────

.PHONY: clean run run-tonal run-vocal stop

# ── CLEAN ───────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Nettoyage des artefacts générés..."
	@# MLflow, outputs, viz
	rm -rf $(CLEAN_DIRS)
	rm -f  $(CLEAN_FILES)
	@# Données intermédiaires / modèles (garde les dossiers vides)
	find data/02_intermediate -type f -delete 2>/dev/null || true
	find data/05_model_input  -type f -delete 2>/dev/null || true
	find data/06_models       -type f -delete 2>/dev/null || true
	@# Python cache
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc'      -delete          2>/dev/null || true
	rm -rf $(EGG_INFO)
	@echo "✅ Nettoyage terminé. Les raw data sont intactes."

# ── RUN (tous les pipelines) ───────────────────────────────────────────────
run: run-tonal run-vocal
	@echo ""
	@echo "🚀 Tous les pipelines terminés. Démarrage des serveurs..."
	@# Kedro-viz en arrière-plan
	kedro viz --no-browser &
	@echo "📊 Kedro-Viz → http://localhost:4141"
	@# MLflow UI en arrière-plan
	mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "📈 MLflow UI  → http://localhost:5000"
	@echo ""
	@echo "✅ Tout est lancé. Utilisez 'make stop' pour arrêter les serveurs."

# ── RUN TONAL ──────────────────────────────────────────────────────────────
run-tonal:
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  Pipeline TONAL"
	@echo "═══════════════════════════════════════"
	kedro run --pipeline ingestion
	kedro run --pipeline features
	kedro run --pipeline training
	kedro run --pipeline evaluation
	@echo "✅ Pipelines tonaux terminés."

# ── RUN VOCAL ──────────────────────────────────────────────────────────────
run-vocal:
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  Pipeline VOCAL"
	@echo "═══════════════════════════════════════"
	kedro run --pipeline ingestion_vocal
	kedro run --pipeline features_vocal
	kedro run --pipeline training_vocal
	kedro run --pipeline evaluation_vocal
	@echo "✅ Pipelines vocaux terminés."

# ── STOP (arrêter les serveurs) ────────────────────────────────────────────
stop:
	@echo "🛑 Arrêt des serveurs..."
	@-pkill -f "kedro viz"   2>/dev/null || true
	@-pkill -f "mlflow ui"   2>/dev/null || true
	@-pkill -f "mlflow server" 2>/dev/null || true
	@echo "✅ Serveurs arrêtés."
