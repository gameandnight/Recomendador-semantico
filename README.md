# Recomendador Semántico — README

**Estado:** versión final probada en local (FastAPI backend + Vite frontend).
Este README describe cómo ejecutar, configurar, tunear y desplegar la versión que tienes funcionando localmente.

---

# 📁 Estructura del proyecto (resumen útil)

```
recomendador-embeddings/
├─ backend/
│  ├─ app/
│  │  └─ main.py          # API FastAPI (buscador híbrido)
│  ├─ data/
│  │  ├─ faiss.index      # índice FAISS (vector)
│  │  └─ meta.pkl         # metadata / productos (list o dict)
│  ├─ evaluate_with_evalset.py
│  └─ evaluate_with_evalset_rerank.py
├─ frontend/               # app Vite (http://localhost:5173)
└─ .venv/                  # entorno virtual (opcional)
```

---

# 🧰 Requisitos

- Python 3.10+ (tu entorno usa 3.10)
- pip packages: `sentence-transformers`, `faiss-cpu` (o `faiss` si procede), `rank-bm25`, `nltk`, `fastapi`, `uvicorn`, `requests`, `tqdm`, etc.
- Node.js + npm para el frontend (Vite).

Instalación básica (backend):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1    # PowerShell (Windows)
pip install -r backend/requirements.txt  # si tienes requirements
# o instalar paquetes manualmente:
pip install sentence-transformers faiss-cpu rank-bm25 nltk fastapi uvicorn requests tqdm
```

Frontend:

```bash
cd frontend
npm install
npm run dev   # abre http://localhost:5173
```

---

# ⚙️ Variables de entorno recomendadas (ejemplos)

Ponlas en PowerShell antes de arrancar (o en `.env` si usas un cargador).

```powershell
$env:FAISS_INDEX_PATH="backend/data/faiss.index"
$env:FAISS_META_PATH="backend/data/meta.pkl"
$env:MIN_SCORE_INDIVIDUAL="0.07"
$env:MIN_FINAL_SCORE="0.045"
$env:SOFTMAX_TEMP="0.30"
$env:OVERLAP_WEIGHT="0.22"
$env:MIN_OVERLAP_FOR_BOOST="0.06"
$env:STEM_BOOST="0.20"
$env:MIN_JACCARD_KEEP="0.08"
$env:USE_RERANKER="1"        # "1" para activar re-ranker
$env:RERANK_K="20"
$env:RERANK_BLEND="0.7"
$env:RERANK_MIN_SCORE="0.045"
```

> Estos valores vienen de pruebas que funcionaron bien en tu entorno. Ajústalos si quieres ser más o menos restrictivo.

---

# ▶️ Ejecutar en local (backend)

Desde la raíz del repo:

```bash
# activar .venv, setear env vars
python -m uvicorn backend.app.main:app --reload --port 8000
```
Frontend:

```bash
cd frontend
npm run dev    # http://localhost:5173
```

---

# 📡 API — Endpoint principal

`POST /search` JSON body:

```json
{
  "q": "patata 3 kg",
  "k": 6,
  "alpha": 0.7,
  "beta": 0.3,
  "use_reranker": true,
  "rerank_k": 20
}
```

Respuesta (ejemplo abreviado):

```json
{
  "query": "patata 3 kg",
  "results": [
    {
      "score": 0.80234,
      "score_pct": 80.23,
      "product": {
        "id": "...",
        "title": "Patata trumferoles Potatum 3 Kg",
        "description": "...",
        "price": 3.99,
        "category": "...",
        "image": "..."
      },
      "source": "textual" // "semantic", "textual" o "hybrid"
    },
    ...
  ]
}
```

Parámetros relevantes:
- `alpha` = peso embeddings/FAISS (semantic)
- `beta` = peso BM25 (textual)
- `use_reranker`: activa re-ranker (cross-encoder)
- `k`: nº de resultados devueltos

---

# 🔎 Comportamiento y lógica (resumen técnico)

- **Candidatos**: unión de top-k FAISS (vectors) y top-k BM25.
- **Normalización**: scores FAISS y BM25 se normalizan (min-max) antes de mezclar.
- **Boost**: coincidencia textual (jaccard) y stemming (NLTK Snowball) pueden aportar boost.
- **Filtros**: se descartan candidatos si la señal individual máxima < `MIN_SCORE_INDIVIDUAL`.
- **Softmax**: se aplica softmax (temperatura `SOFTMAX_TEMP`) a las puntuaciones crudas para obtener `final_score`.
- **Reranking**: opcional, usa `cross-encoder/ms-marco-MiniLM-L-6-v2` (pesado en CPU). Solo se ejecuta si la confianza media de los topk supera `RERANK_MIN_SCORE`.
- **BM25**: si `tokenized_corpus` no está en `meta.pkl`, el servidor generará tokenized_corpus automáticamente desde `products` y arrancará BM25 (rank_bm25).

---

# 🧪 Evaluación (scripts incluidos)

Has probado y usado estos scripts; aquí los comandos tal cual:

Evaluación estándar:
```bash
python backend/evaluate_with_evalset.py --eval backend/eval_queries_with_gtids.json --steps 11 --k 6
```

Evaluación con reranker:
```bash
python backend/evaluate_with_evalset_rerank.py --eval backend/eval_queries_with_gtids.json --steps 11 --k 6 --rerank_k 20
```

Notas:
- Los scripts hacen peticiones HTTP a `http://localhost:8000/search`. Asegúrate de que el backend esté UP antes de lanzar la evaluación.
- Si ves errores `500` en estas pruebas, revisa los logs del backend (a menudo son por FAISS shapes, re-ranker o falta de assets).

---

# 🛠 Troubleshooting (problemas frecuentes y soluciones)

- **CORS / preflight 405**: asegúrate que `CORS_ORIGINS` en `main.py` incluye `http://localhost:5173` o usa `allow_origins=["*"]` para desarrollo (no recomendado en prod).
- **`ValueError: too many values to unpack (expected 2)` en FAISS.search**: pasa un `np.array([q_vec], dtype="float32")` con forma (1, D). Si ves ese error, revisa que `q_vec` sea un vector 1D y que `faiss_index` esté cargado.
- **Re-ranker muy lento / memoria**: en CPU puede tardar; pon `USE_RERANKER=0` para pruebas ligeras.
- **BM25 desactivado**: si ves mensaje `tokenized_corpus vacío -> BM25 desactivado`, o bien crea `tokenized_corpus` dentro de `meta.pkl` o deja que el servidor lo genere automáticamente (si `products` están bien formateados).
- **Resultados poco relevantes**: afina `MIN_SCORE_INDIVIDUAL`, `MIN_FINAL_SCORE`, `SOFTMAX_TEMP`, `OVERLAP_WEIGHT`, `MIN_OVERLAP_FOR_BOOST`. Valores recomendados (a partir de pruebas): ver sección Variables de entorno.
- **Querys “gibberish”**: el servidor marca consultas sin tokens útiles (p. ej. "de", "la" o cadenas de símbolos) y devuelve sin resultados — esto es intencional.

---


