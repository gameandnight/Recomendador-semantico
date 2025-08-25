"""
Script para construir el índice FAISS y guardar el corpus tokenizado para BM25.
Uso:
    python backend/build_index.py
"""
import json, os
from app.embeddings import EmbeddingModel
from app.faiss_index import FaissIndex
import pickle
import re

DATA_PATH = os.environ.get("PRODUCTS_PATH", "data/products.json")
OUT_INDEX_DIR = os.environ.get("OUT_INDEX_DIR", "backend/data")
os.makedirs(OUT_INDEX_DIR, exist_ok=True)

def tokenize(text):
    # tokenización simple: lowercase, quitar puntuación básica, split
    text = text.lower()
    text = re.sub(r'[^a-z0-9áéíóúüñ\s]', ' ', text)
    tokens = text.split()
    return tokens

def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit("No existe data/products.json. Ejecuta scripts/generate_products.py")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    texts = [p.get("title","") + " . " + p.get("description","") for p in products]
    emb_model = EmbeddingModel()
    emb = emb_model.embed_texts(texts)
    dim = emb.shape[1]
    idx = FaissIndex(dim=dim)
    idx.add(emb, products)
    # Save index and metadata into backend/data
    idx.save()  # this uses index path default "data/faiss.index" unless env set

    # Additionally save tokenized corpus for BM25 into backend/data/corpus_tokens.pkl
    corpus_tokens = [tokenize(t) for t in texts]
    token_path = os.path.join(OUT_INDEX_DIR, "corpus_tokens.pkl")
    with open(token_path, "wb") as f:
        pickle.dump(corpus_tokens, f)
    # Also copy (or save) products metadata into backend/data/products_meta.json for backend use
    meta_path = os.path.join(OUT_INDEX_DIR, "products_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print("Índice construido con", len(products), "vectores.")
    print("Tokenized corpus y products_meta guardados en", OUT_INDEX_DIR)

if __name__ == "__main__":
    main()
