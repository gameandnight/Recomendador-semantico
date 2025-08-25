# Dockerfile (backend)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

# system deps for faiss + torch + sentence-transformers (básico)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates python3-dev \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements (mejor tener requirements.txt generado)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copia código
COPY backend /app/backend

# Directorio para datos persistentes (Render mountará aquí)
VOLUME ["/data"]
ENV FAISS_INDEX_PATH="/data/faiss.index"
ENV FAISS_META_PATH="/data/meta.pkl"

# Variables por defecto que puedes overridear en Render
ENV MIN_SCORE_INDIVIDUAL=0.07 \
    MIN_FINAL_SCORE=0.045 \
    SOFTMAX_TEMP=0.30 \
    OVERLAP_WEIGHT=0.22 \
    MIN_OVERLAP_FOR_BOOST=0.06 \
    STEM_BOOST=0.20 \
    MIN_JACCARD_KEEP=0.08 \
    USE_RERANKER=1 \
    RERANK_K=20 \
    RERANK_BLEND=0.7 \
    RERANK_MIN_SCORE=0.045

EXPOSE 8000

# Comando de arranque (uvicorn)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
