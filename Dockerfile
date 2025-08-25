# Dockerfile — Opción ligera para entornos con poca RAM (prueba)
FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app

# Copiamos environment para cache
COPY environment.yml /tmp/environment.yml

# Instala mamba/conda env (faiss/pytorch desde conda)
RUN conda install -y -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml -n reco-env && \
    conda clean -afy

# asegúrate de usar el entorno
ENV PATH=/opt/conda/envs/reco-env/bin:$PATH

# Copia el repo
COPY . /app
WORKDIR /app

# Puerto documentativo (Render usa $PORT)
ENV PORT=8000

# REDUCE MEMORIA: desactivar re-ranker por defecto (puedes cambiar en Settings de Render)
ENV USE_RERANKER=0

# Start con uvicorn (menos overhead que gunicorn + workers)
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
