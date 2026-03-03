# ============================================================================
#  Makefile — audiogram-mlops (Kedro + MLflow)
# ============================================================================
#  Usage :
#    make clean        → supprime tous les artefacts générés
#    make tonal        → pipelines tonaux (sans Optuna)
#    make vocal        → pipelines vocaux (sans Optuna)
#    make run          → tonal + vocal (sans Optuna)
#    make tune-tonal   → pipelines tonaux + Optuna
#    make tune-vocal   → pipelines vocaux + Optuna
#    make tune         → tonal + vocal + Optuna
#    make viz          → lance Kedro-Viz
#    make mlflow       → lance MLflow UI
#    make stop         → arrête les serveurs
# ============================================================================

SHELL := /bin/bash

.PHONY: clean tonal vocal run tune-tonal tune-vocal tune viz mlflow stop

# ── CLEAN ───────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Nettoyage des artefacts générés..."
	rm -rf mlruns outputs .viz
	rm -f  mlflow.db info.log
	find data/02_intermediate -type f -delete 2>/dev/null || true
	find data/05_model_input  -type f -delete 2>/dev/null || true
	find data/06_models       -type f -delete 2>/dev/null || true
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc'      -delete          2>/dev/null || true
	rm -rf src/audiogram_mlops.egg-info
	@echo "✅ Nettoyage terminé. Les raw data sont intactes."

# ── SANS OPTUNA ─────────────────────────────────────────────────────────────
tonal:
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  Pipeline TONAL (sans Optuna)"
	@echo "═══════════════════════════════════════"
	kedro run --pipeline ingestion
	kedro run --pipeline features
	kedro run --pipeline training
	kedro run --pipeline evaluation
	@echo "✅ Pipelines tonaux terminés."

vocal:
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  Pipeline VOCAL (sans Optuna)"
	@echo "═══════════════════════════════════════"
	kedro run --pipeline ingestion_vocal
	kedro run --pipeline features_vocal
	kedro run --pipeline training_vocal
	kedro run --pipeline evaluation_vocal
	@echo "✅ Pipelines vocaux terminés."

run: tonal vocal

# ── AVEC OPTUNA ─────────────────────────────────────────────────────────────
tune-tonal:
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  Pipeline TONAL + Optuna"
	@echo "═══════════════════════════════════════"
	kedro run --pipeline ingestion
	kedro run --pipeline features
	python tune_model.py
	@echo "✅ Optimisation tonale terminée."

tune-vocal:
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  Pipeline VOCAL + Optuna"
	@echo "═══════════════════════════════════════"
	kedro run --pipeline ingestion_vocal
	kedro run --pipeline features_vocal
	python tune_model_vocal.py
	@echo "✅ Optimisation vocale terminée."

tune: tune-tonal tune-vocal

# ── SERVEURS ────────────────────────────────────────────────────────────────
viz:
	@echo "📊 Kedro-Viz → http://localhost:4141"
	kedro viz

mlflow:
	@-pkill -f "mlflow ui" 2>/dev/null || true
	@-pkill -f "mlflow server" 2>/dev/null || true
	@sleep 1
	@echo "📈 MLflow UI → http://localhost:5000"
	mlflow ui --host 0.0.0.0 --port 5000

stop:
	@echo "🛑 Arrêt des serveurs..."
	@-pkill -f "kedro viz"     2>/dev/null || true
	@-pkill -f "mlflow ui"    2>/dev/null || true
	@-pkill -f "mlflow server" 2>/dev/null || true
	@echo "✅ Serveurs arrêtés."
