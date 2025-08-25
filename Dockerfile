# Dockerfile — usa conda/mamba para instalar binarios (faiss, torch, tokenizers, etc.)
FROM continuumio/miniconda3:latest

# Evitamos prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

# Copiamos el environment para aprovechar cache
COPY environment.yml /tmp/environment.yml

# Instala mamba (más rápido que conda) y crea el entorno
RUN conda install -y -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml -n reco-env && \
    conda clean -afy

# Aseguramos que el entorno esté disponible en PATH (usa conda run en CMD)
SHELL ["conda", "run", "-n", "reco-env", "/bin/bash", "-lc"]

# Copiamos el resto del repo
COPY . /app
WORKDIR /app

# Expone el puerto (documentativo, Render usa $PORT igualmente)
EXPOSE 8000

# Comando por defecto para arrancar la app con gunicorn + uvicorn worker
# Usa el puerto de Render ($PORT) o 8000 por defecto en local
CMD ["conda", "run", "-n", "reco-env", "bash", "-c", "gunicorn -k uvicorn.workers.UvicornWorker backend.app.main:app --bind 0.0.0.0:${PORT:-8000} --workers 1"]
