# =============================================================================
#  Dockerfile — audiogram-mlops API REST
# =============================================================================
#  Build  : docker build -t audiogram-mlops:latest .
#  Run    : docker run -p 5001:5001 audiogram-mlops:latest
# =============================================================================

FROM python:3.10-slim

# Évite les invites interactives et les fichiers .pyc
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dépendances système ─────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source du projet ───────────────────────────────────────────────────
COPY . .

# ── Installation du package Kedro en mode éditable ──────────────────────────
RUN pip install --no-cache-dir -e .

# ── Répertoire de données API ───────────────────────────────────────────────
RUN mkdir -p data/07_api_inputs

# ── Port exposé ─────────────────────────────────────────────────────────────
EXPOSE 5001

# ── Commande de démarrage ───────────────────────────────────────────────────
# Lance l'API Flask (app.py) — pour la production, utiliser gunicorn
CMD ["python", "app.py"]
