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

# Aseguramos que el entorno esté disponible en PATH
ENV PATH=/opt/conda/envs/reco-env/bin:$PATH

# Copiamos el resto del repo
COPY . /app
WORKDIR /app

# Puerto documentativo (Render usa $PORT)
EXPOSE 8000

# Asegurar que Render conozca un valor por defecto si no hay $PORT
ENV PORT=8000

# Comando de arranque:
# - exec para recibir señales correctamente
# - --timeout aumentado (ej. 300s) para evitar worker timeouts mientras cargan modelos
# - --preload para cargar la app en el master (reduce la sobrecarga por fork)
# - logs a stdout/stderr para que Render los muestre
CMD ["sh", "-c", "exec gunicorn -k uvicorn.workers.UvicornWorker backend.app.main:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 300 --preload --log-level info --access-logfile - --error-logfile -"]
