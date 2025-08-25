# backend/evaluate_with_evalset_rerank.py
import json
import time
import requests
import os
from statistics import mean

API = os.environ.get("REC_API", "http://localhost:8000/search")
EVAL_PATH = "backend/eval_queries.json"
OUT_PATH = "backend/eval_results_rerank.json"

K = 5  # recall@K
TIMEOUT = 30

def call_search(q, k=K, alpha=0.7, beta=0.3, use_reranker=True, rerank_k=50):
    payload = {"q": q, "k": k, "alpha": alpha, "beta": beta, "use_reranker": use_reranker, "rerank_k": rerank_k}
    r = requests.post(API, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def recall_at_k_single(resp_json, gt_ids, k=K):
    results = resp_json.get("results", [])
    top_ids = [r["product"]["id"] for r in results[:k]]
    return 1 if any(x in top_ids for x in gt_ids) else 0

def reciprocal_rank_single(resp_json, gt_ids):
    results = resp_json.get("results", [])
    for i, r in enumerate(results, start=1):
        if r["product"]["id"] in gt_ids:
            return 1.0 / i
    return 0.0

def evaluate_for_combo(queries, alpha, beta, k=K, rerank_k=50):
    recalls = []
    recalls1 = []
    mrrs = []
    t0 = time.time()
    for q in queries:
        try:
            resp = call_search(q["q"], k=k, alpha=alpha, beta=beta, use_reranker=True, rerank_k=rerank_k)
            r_at_k = recall_at_k_single(resp, q["gt_ids"], k=k)
            r_at_1 = recall_at_k_single(resp, q["gt_ids"], k=1)
            mrr = reciprocal_rank_single(resp, q["gt_ids"])
        except Exception as e:
            print("Error query:", q["q"][:60], e)
            r_at_k = 0
            r_at_1 = 0
            mrr = 0
        recalls.append(r_at_k)
        recalls1.append(r_at_1)
        mrrs.append(mrr)
    elapsed = time.time() - t0
    return {
        "alpha": alpha,
        "beta": beta,
        "recall@{}".format(k): mean(recalls),
        "recall@1": mean(recalls1),
        "mrr": mean(mrrs),
        "time_s": elapsed,
        "n_queries": len(queries)
    }

def grid_search(queries, steps=11, rerank_k=50):
    grid = [i/(steps-1) for i in range(steps)]  # 0.0 .. 1.0
    results = []
    for a in grid:
        b = round(1.0 - a, 6)
        print(f"Evaluating alpha={a:.2f} beta={b:.2f} (rerank_k={rerank_k}) ...")
        stats = evaluate_for_combo(queries, a, b, k=K, rerank_k=rerank_k)
        print(f" -> recall@{K}: {stats['recall@{}'.format(K)]:.3f} recall@1: {stats['recall@1']:.3f} mrr: {stats['mrr']:.3f} time: {stats['time_s']:.1f}s")
        results.append(stats)
    results.sort(key=lambda x: (x[f"recall@{K}"], x["mrr"]), reverse=True)
    return results

if __name__ == "__main__":
    if not os.path.exists(EVAL_PATH):
        raise SystemExit(f"No existe {EVAL_PATH}. Genera eval set primero.")
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print("Queries a evaluar:", len(queries))
    # ajusta rerank_k aquí si quieres (p. ej. 20 o 50)
    rerank_k = int(os.environ.get("RERANK_K", "50"))
    res = grid_search(queries, steps=11, rerank_k=rerank_k)
    print("\nTop 5 combos:")
    for r in res[:5]:
        print(f"alpha={r['alpha']:.2f} beta={r['beta']:.2f} recall@{K}={r[f'recall@{K}']:.3f} recall@1={r['recall@1']:.3f} mrr={r['mrr']:.3f}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nResultados guardados en", OUT_PATH)
