# Dockerfile (usa conda / mamba para instalar faiss/pytorch binarios)
FROM continuumio/miniconda3:23.11.1-1

# Evitamos prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

# Copiamos archivos necesarios primero (caching)
COPY environment.yml /tmp/environment.yml
# Instala mamba (más rápido que conda) y crea el entorno
RUN conda install -y -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml && \
    conda clean -afy

# Aseguramos que el entorno esté disponible en PATH
SHELL ["conda", "run", "-n", "reco-env", "/bin/bash", "-lc"]

# Copiamos el resto del repo
COPY . /app
WORKDIR /app

# Expone el puerto (uvicorn)
EXPOSE 8000

# Comando por defecto para arrancar la app
CMD ["conda", "run", "-n", "reco-env", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

