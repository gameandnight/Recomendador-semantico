# Dockerfile — usa conda/mamba para instalar binarios (faiss, torch, tokenizers, etc.)
FROM continuumio/miniconda3:23.11.1-1

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app

# Instala utilidades del sistema necesarias para algunos paquetes y para mamba
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential swig git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copiamos environment.yml e instalamos mamba y el env
COPY environment.yml /tmp/environment.yml
RUN conda install -y -n base -c conda-forge mamba \
    && mamba env create -f /tmp/environment.yml -n reco-env \
    && conda clean -afy

# Aseguramos que el entorno esté disponible (usaremos conda run)
SHELL ["conda", "run", "-n", "reco-env", "/bin/bash", "-lc"]

# Copiamos el resto del repo (incluye backend/, frontend/ y backend/data/)
COPY . /app
WORKDIR /app

# Pre-instala paquetes pip restantes dentro del entorno (si usas requirements.txt)
# (Opcional si ya están en environment.yml — pero es seguro tenerlo)
RUN if [ -f backend/requirements.txt ]; then pip install -r backend/requirements.txt; fi

# Expone el puerto que usa la app
EXPOSE 8000

# Command por defecto: usa gunicorn con uvicorn workers (producción ligera)
CMD ["conda", "run", "-n", "reco-env", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "backend.app.main:app", "--bind", "0.0.0.0:8000", "--workers", "1"]


