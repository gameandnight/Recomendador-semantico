"""
backend/app/main.py
Versión parcheada: mantiene la lógica original pero añade:
 - stopwords (nltk) y tokenización que las filtra
 - generación automática de tokenized_corpus desde products para activar BM25
 - defensas robustas en faiss.search (shape/dtype)
 - manejo seguro del reranker y normalización de salidas
 - logs claros y mínimos cambios en scoring/umbral
"""

import os
import json
import logging
import math
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# externos (asegúrate de tenerlos instalados)
try:
    import faiss
except Exception:
    faiss = None

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# BM25 (opcional)
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# nltk para stemming y stopwords
try:
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:
    nltk = None
    SnowballStemmer = None
    nltk_stopwords = None

logger = logging.getLogger("backend.app.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# -----------------------------------------------------------------------------
# Config desde ENV (parámetros afinables)
# -----------------------------------------------------------------------------
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "backend/data/faiss.index")
FAISS_META_PATH = os.getenv("FAISS_META_PATH", "backend/data/meta.pkl")

MIN_SCORE_INDIVIDUAL = float(os.getenv("MIN_SCORE_INDIVIDUAL", "0.06"))
MIN_FINAL_SCORE = float(os.getenv("MIN_FINAL_SCORE", "0.035"))
MIN_OVERLAP_FOR_BOOST = float(os.getenv("MIN_OVERLAP_FOR_BOOST", "0.01"))
OVERLAP_WEIGHT = float(os.getenv("OVERLAP_WEIGHT", "0.15"))
STEM_BOOST = float(os.getenv("STEM_BOOST", "0.20"))
SOFTMAX_TEMP = float(os.getenv("SOFTMAX_TEMP", "0.4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", str(MIN_FINAL_SCORE)))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", str(MIN_FINAL_SCORE)))
MIN_JACCARD_KEEP = float(os.getenv("MIN_JACCARD_KEEP", "0.02"))

USE_RERANKER = os.getenv("USE_RERANKER", "0") in ("1", "true", "True")
RERANK_K = int(os.getenv("RERANK_K", "20"))
RERANK_BLEND = float(os.getenv("RERANK_BLEND", "0.7"))

# CORS: front dev
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# -----------------------------------------------------------------------------
# FastAPI app + CORS middleware
# -----------------------------------------------------------------------------
app = FastAPI(title="Recomendador Semántico - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,      # para desarrollo. Si quieres permitir todo: ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class SearchRequest(BaseModel):
    q: str
    k: Optional[int] = 6
    alpha: Optional[float] = 0.7   # peso embeddings
    beta: Optional[float] = 0.3    # peso bm25
    use_reranker: Optional[bool] = False
    rerank_k: Optional[int] = None

# -----------------------------------------------------------------------------
# Helpers: tokenización ligera, stemming, jaccard, softmax...
# -----------------------------------------------------------------------------
# TOKEN regex
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Intentar preparar stopwords (descarga si hace falta)
STOPWORDS = set()
if nltk and nltk_stopwords:
    try:
        # si no están descargadas, intentar descargar de forma silenciosa
        try:
            _ = nltk_stopwords.words("spanish")
        except Exception:
            try:
                nltk.download("stopwords", quiet=True)
            except Exception:
                pass
        STOPWORDS = set(nltk_stopwords.words("spanish"))
    except Exception:
        STOPWORDS = set()

logger.info("NLTK disponible: %s  SnowballStemmer: %s  stopwords_loaded: %s",
            bool(nltk), bool(SnowballStemmer), bool(STOPWORDS))

def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize básico; opcionalmente elimina stopwords (español)."""
    if not text:
        return []
    toks = TOKEN_RE.findall(str(text).lower())
    if remove_stopwords and STOPWORDS:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks

# Stemming (Snowball spanish) si está disponible
stemmer = None
if SnowballStemmer:
    try:
        stemmer = SnowballStemmer("spanish")
    except Exception:
        stemmer = None

def stem_tokens(tokens: List[str]) -> List[str]:
    if not stemmer:
        return tokens
    return [stemmer.stem(t) for t in tokens]

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    uni = a.union(b)
    return len(inter) / len(uni)

def softmax(scores: List[float], temp: float = 1.0) -> List[float]:
    if not scores:
        return []
    scaled = [s / (temp if temp > 0 else 1e-6) for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    ssum = sum(exps)
    if ssum == 0:
        return [1.0 / len(exps)] * len(exps)
    return [e / ssum for e in exps]

def normalize_minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [0.0 if hi == 0 else 1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

# -----------------------------------------------------------------------------
# Carga assets: meta, faiss index, bm25 (si está)
# -----------------------------------------------------------------------------
meta = {}
products: List[Dict[str, Any]] = []
faiss_index = None
emb_model = None
reranker = None
bm25 = None
tokenized_corpus: List[List[str]] = []

def load_assets():
    global meta, products, faiss_index, emb_model, reranker, bm25, tokenized_corpus

    # carga meta (pickle). Puede ser un dict con keys o una lista simple
    if os.path.exists(FAISS_META_PATH):
        logger.info("Cargando assets desde %s", FAISS_META_PATH)
        try:
            with open(FAISS_META_PATH, "rb") as f:
                data = pickle.load(f)
            # Diferentes formatos posibles:
            if isinstance(data, dict):
                products = data.get("products", data.get("products_meta", []))
                tokenized_corpus = data.get("tokenized_corpus", []) or []
            elif isinstance(data, list):
                # formato simple: lista de productos
                products = data
                tokenized_corpus = []  # lo generaremos abajo si hace falta
                logger.info("meta.pkl contiene una lista de productos (len=%d).", len(products))
            else:
                # objeto inesperado
                logger.warning("Formato inesperado en meta.pkl (%s). Intentando usar como lista.", type(data))
                try:
                    products = list(data)
                except Exception:
                    products = []
            meta["n_products"] = len(products)
            logger.info("Meta cargada: productos=%s", meta["n_products"])
        except Exception as e:
            logger.exception("Error leyendo FAISS_META_PATH: %s", e)
    else:
        logger.warning("Meta no encontrada en %s. Esperando faiss + products.json", FAISS_META_PATH)

    # Si tokenized_corpus está vacío, lo generamos desde products (para activar BM25)
    if not tokenized_corpus and products:
        logger.info("Generando tokenized_corpus automáticamente a partir de products (para BM25)...")
        tokenized_corpus = []
        for p in products:
            title = str(p.get("title", "") or "")
            desc = str(p.get("description", "") or "")
            toks = tokenize(title + " " + desc, remove_stopwords=True)
            toks = stem_tokens(toks)
            tokenized_corpus.append(toks)
        logger.info("tokenized_corpus generado (len=%d).", len(tokenized_corpus))

    # carga FAISS
    if faiss is None:
        logger.warning("faiss no disponible en este entorno. Instala faiss-cpu si quieres búsqueda por vectores.")
    else:
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                idx = faiss.read_index(FAISS_INDEX_PATH)
                globals()["faiss_index"] = idx
                logger.info("FAISS index cargado desde %s", FAISS_INDEX_PATH)
            except Exception as e:
                logger.exception("No se pudo leer FAISS index: %s", e)
                globals()["faiss_index"] = None
        else:
            logger.warning("FAISS index no encontrado en %s", FAISS_INDEX_PATH)

    # embeddings
    try:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        emb_model_local = SentenceTransformer(model_name)
        globals()["emb_model"] = emb_model_local
        logger.info("Modelo de embeddings cargado: %s", emb_model_local.__class__.__name__)
    except Exception as e:
        logger.exception("Error cargando SentenceTransformer: %s", e)
        globals()["emb_model"] = None

    # reranker (cross encoder) opcional
    if USE_RERANKER:
        try:
            model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            if CrossEncoder:
                reranker_local = CrossEncoder(model_name)
                globals()["reranker"] = reranker_local
                logger.info("Re-ranker cargado: %s", model_name)
            else:
                logger.warning("CrossEncoder no disponible en el entorno.")
                globals()["reranker"] = None
        except Exception as e:
            logger.exception("No se pudo cargar reranker: %s", e)
            globals()["reranker"] = None

    # BM25 si tenemos tokenized_corpus y rank_bm25 instalado
    if BM25Okapi and tokenized_corpus:
        try:
            bm25_local = BM25Okapi(tokenized_corpus)
            globals()["bm25"] = bm25_local
            logger.info("BM25 preparado (desde tokenized_corpus).")
        except Exception as e:
            logger.exception("Error iniciando BM25: %s", e)
            globals()["bm25"] = None
    else:
        if not BM25Okapi:
            logger.info("rank_bm25 no instalado -> BM25 desactivado.")
        elif not tokenized_corpus:
            logger.info("tokenized_corpus vacío -> BM25 desactivado (si quieres BM25, crea tokenized_corpus en meta.pkl).")

# carga en startup
load_assets()

# -----------------------------------------------------------------------------
# Funciones core de búsqueda: faiss_search, bm25_search, union, scoring...
# -----------------------------------------------------------------------------
def faiss_search_vec(q_vec, k: int = 50) -> Tuple[List[int], List[float]]:
    """
    Asegura que q_vec sea un np.array float32 con shape (1, d) antes de llamar a faiss.
    Devuelve ids y scores (similares).
    En caso de error, no lanza excepción, devuelve listas vacías.
    """
    if faiss is None or globals().get("faiss_index") is None:
        return [], []
    try:
        import numpy as np
        vec = np.asarray(q_vec, dtype="float32")
        # reshape defensivo: si es 1D, convertir a (1, d)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        elif vec.ndim > 2:
            # aplana todo salvo la última dimensión
            vec = vec.reshape(1, -1)
        # ensure contiguous
        vec = np.ascontiguousarray(vec, dtype="float32")
        D, I = globals()["faiss_index"].search(vec, k)
        ids = I[0].tolist() if len(I) > 0 else []
        dists = D[0].tolist() if len(D) > 0 else []
        sims = [float(x) for x in dists]
        return ids, sims
    except Exception as e:
        logger.exception("FAISS search error: %s", e)
        return [], []

def bm25_search(q_tokens: List[str], k: int = 50) -> Tuple[List[int], List[float]]:
    if globals().get("bm25") is None:
        return [], []
    scores = globals()["bm25"].get_scores(q_tokens)
    import numpy as np
    if len(scores) == 0:
        return [], []
    idx = np.argsort(scores)[::-1][:k]
    return idx.tolist(), [float(scores[i]) for i in idx]

def build_candidate_union(faiss_ids, bm25_ids, max_candidates=200) -> List[int]:
    seen = set()
    out = []
    for i in faiss_ids:
        if i not in seen:
            seen.add(i); out.append(i)
    for i in bm25_ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out[:max_candidates]

def product_by_index(i: int) -> Optional[Dict[str,Any]]:
    try:
        return products[i]
    except Exception:
        return None

def compute_overlap_boost(q_tokens_set: set, prod_tokens_set: set, q_stems_set: set, prod_stems_set: set) -> float:
    j = jaccard(q_tokens_set, prod_tokens_set)
    stem_j = jaccard(q_stems_set, prod_stems_set)
    boost = 0.0
    if j >= MIN_OVERLAP_FOR_BOOST:
        boost += OVERLAP_WEIGHT * j
    if stem_j >= MIN_OVERLAP_FOR_BOOST:
        boost += STEM_BOOST * stem_j
    return boost

# -----------------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "index_size": meta.get("n_products", len(products))}

# -----------------------------------------------------------------------------
# Search endpoint
# -----------------------------------------------------------------------------
@app.post("/search")
async def search(req: Request, body: SearchRequest):
    q = body.q.strip()
    if not q:
        return {"query": q, "results": []}

    # tokenizamos eliminando stopwords para evitar que 'de','la','y' influyan demasiado
    q_tokens = tokenize(q, remove_stopwords=True)
    if not q_tokens:
        logger.info("Query marcada como gibberish (tokens vacíos tras stopwords): %s", q)
        return {"query": q, "results": []}
    q_stems = stem_tokens(q_tokens)

    if globals().get("emb_model") is None:
        raise HTTPException(status_code=500, detail="Embeddings model not loaded")

    # encode: preferimos string completo si sólo 1 token, como antes
    try:
        input_for_encode = q_tokens if len(q_tokens) > 1 else q
        # algunas versiones no tienen normalize_embeddings -> atrapamos TypeError
        try:
            q_vec = globals()["emb_model"].encode(input_for_encode, convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            q_vec = globals()["emb_model"].encode(input_for_encode, convert_to_numpy=True)
    except Exception as e:
        logger.exception("Error encoding query: %s", e)
        raise HTTPException(status_code=500, detail="Error computing embeddings")

    # asegurar shape/dtype
    import numpy as np
    q_vec = np.asarray(q_vec, dtype="float32")
    if q_vec.ndim == 1:
        q_vec_reshaped = q_vec.reshape(1, -1)
    else:
        q_vec_reshaped = q_vec

    faiss_k = max(body.k * 6, 50)
    faiss_ids, faiss_scores_raw = faiss_search_vec(q_vec_reshaped, k=faiss_k)
    bm25_k = max(body.k * 6, 50)
    bm25_ids, bm25_scores_raw = bm25_search(q_tokens, k=bm25_k) if globals().get("bm25") is not None else ([], [])

    norm_faiss = normalize_minmax(faiss_scores_raw) if faiss_scores_raw else []
    norm_bm25 = normalize_minmax(bm25_scores_raw) if bm25_scores_raw else []

    faiss_map = {pid: norm_faiss[i] for i, pid in enumerate(faiss_ids)} if norm_faiss else {}
    bm25_map = {pid: norm_bm25[i] for i, pid in enumerate(bm25_ids)} if norm_bm25 else {}

    union_cands = build_candidate_union(faiss_ids, bm25_ids, max_candidates=200)
    logger.debug("CANDIDATES DEBUG: faiss_candidates=%d bm25_candidates=%d union_candidates=%d",
                 len(faiss_ids), len(bm25_ids), len(union_cands))

    results = []
    q_tokens_set = set(q_tokens)
    q_stems_set = set(q_stems)

    for idx in union_cands:
        prod = product_by_index(idx)
        if prod is None:
            continue
        text = " ".join([str(prod.get("title","")), str(prod.get("description",""))])
        # tokenizamos producto también eliminando stopwords
        prod_tokens = tokenize(text, remove_stopwords=True)
        prod_stems = stem_tokens(prod_tokens)

        prod_tokens_set = set(prod_tokens)
        prod_stems_set = set(prod_stems)

        faiss_score = faiss_map.get(idx, 0.0)
        bm25_score = bm25_map.get(idx, 0.0)

        # filtro por señal individual
        if max(faiss_score, bm25_score) < MIN_SCORE_INDIVIDUAL:
            continue

        boost = compute_overlap_boost(q_tokens_set, prod_tokens_set, q_stems_set, prod_stems_set)

        combined = body.alpha * faiss_score + body.beta * bm25_score + boost

        results.append({
            "idx": idx,
            "product": prod,
            "raw": combined,
            "faiss": faiss_score,
            "bm25": bm25_score,
            "boost": boost,
            "jaccard": jaccard(q_tokens_set, prod_tokens_set)
        })

    if not results:
        logger.info("Query in GENERIC_NO_RESULTS: %s", q)
        return {"query": q, "results": []}

    # keep ones with either good jaccard or some positive boost
    results = [r for r in results if r["jaccard"] >= MIN_JACCARD_KEEP or r["boost"] > 0]

    if not results:
        logger.info("After jaccard/boost filtering -> no results for: %s", q)
        return {"query": q, "results": []}

    raw_scores = [r["raw"] for r in results]
    sm = softmax(raw_scores, temp=SOFTMAX_TEMP)
    for i, r in enumerate(results):
        r["final_score"] = sm[i]

    results.sort(key=lambda x: x["final_score"], reverse=True)

    filtered = [r for r in results if r["final_score"] >= MIN_FINAL_SCORE]

    if not filtered:
        logger.info("Top individual norms low (faiss=%s bm25=%s) < MIN_SCORE_INDIVIDUAL(%s) -> no results.",
                    max([r["faiss"] for r in results]) if results else 0.0,
                    max([r["bm25"] for r in results]) if results else 0.0,
                    MIN_SCORE_INDIVIDUAL)
        return {"query": q, "results": []}

    k = body.k or 6
    topk = filtered[: max(k, RERANK_K if body.use_reranker else k)]

    # Reranker (opcional)
    if body.use_reranker and USE_RERANKER and globals().get("reranker") is not None:
        avg_top = sum([t["final_score"] for t in topk]) / max(1, len(topk))
        if avg_top >= RERANK_MIN_SCORE:
            texts = [t["product"].get("title","") + " " + t["product"].get("description","") for t in topk]
            queries = [q] * len(texts)
            try:
                rerank_out = globals()["reranker"].predict(list(zip(queries, texts)))
                rerank_scores = [float(x) for x in (rerank_out.tolist() if hasattr(rerank_out, "tolist") else rerank_out)]
            except Exception as e:
                logger.exception("Error en reranker.predict: %s", e)
                rerank_scores = []
            if rerank_scores:
                norm_r = normalize_minmax(rerank_scores)
                for i, t in enumerate(topk):
                    blended = RERANK_BLEND * norm_r[i] + (1 - RERANK_BLEND) * t["final_score"]
                    t["final_score"] = blended
                topk.sort(key=lambda x: x["final_score"], reverse=True)

    out_results = []
    for r in topk[:k]:
        prod = r["product"]
        score_pct = float(r["final_score"]) * 100.0
        out_results.append({
            "score": round(r["final_score"], 6),
            "score_pct": round(score_pct, 2),
            "product": {
                "id": prod.get("id"),
                "title": prod.get("title"),
                "description": prod.get("description", ""),
                "price": prod.get("price"),
                "category": prod.get("category"),
                "image": prod.get("image", "")
            },
            "source": "hybrid" if (r["faiss"]>0 and r["bm25"]>0) else ("semantic" if r["faiss"]>r["bm25"] else "textual")
        })

    return {"query": q, "results": out_results}

# -----------------------------------------------------------------------------
# run dev
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
