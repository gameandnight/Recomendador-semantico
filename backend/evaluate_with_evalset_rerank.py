#!/usr/bin/env python3
"""
evaluate_with_evalset_rerank.py
Evaluación usando re-ranker (si está habilitado en backend).
Uso:
    python backend/evaluate_with_evalset_rerank.py --eval backend/eval_queries_with_gtids.json --steps 11 --k 6 --rerank_k 20
"""

import argparse
import json
import time
import requests
import sys
from typing import List, Dict, Any

API = "http://localhost:8000/search"
TIMEOUT = 10

def load_queries(path: str) -> List[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for q in raw:
        entry = {"q": q.get("q") or q.get("query") or q.get("title") or ""}
        if "gt_ids" in q and isinstance(q["gt_ids"], list):
            entry["gt_ids"] = q["gt_ids"]
        elif "pid" in q and q["pid"]:
            entry["gt_ids"] = [q["pid"]]
        elif "id" in q:
            entry["gt_ids"] = [q["id"]]
        else:
            entry["gt_ids"] = []
        out.append(entry)
    return out

def call_search(q: str, k: int, alpha: float, beta: float, use_reranker: bool, rerank_k: int=None) -> Dict[str,Any]:
    payload = {
        "q": q,
        "k": k,
        "alpha": alpha,
        "beta": beta,
        "use_reranker": bool(use_reranker)
    }
    if rerank_k is not None:
        payload["rerank_k"] = int(rerank_k)
    try:
        r = requests.post(API, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Warning: search request failed for q='{q[:80]}' -> {e}", file=sys.stderr)
        return {"query": q, "results": []}

def eval_queries(queries: List[Dict[str,Any]], alpha: float, beta: float, k: int, rerank_k: int):
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    mrr_sum = 0.0
    n = 0
    total_time = 0.0

    for q in queries:
        if not q.get("q"):
            continue
        gt = set(q.get("gt_ids", []))
        if not gt:
            continue
        start = time.time()
        resp = call_search(q["q"], k=k, alpha=alpha, beta=beta, use_reranker=True, rerank_k=rerank_k)
        elapsed = time.time() - start
        total_time += elapsed

        results = resp.get("results", [])
        ids = []
        for r in results:
            p = r.get("product", {})
            pid = p.get("id") or p.get("pid") or p.get("product_id")
            if pid:
                ids.append(str(pid))

        found_rank = None
        for i, pid in enumerate(ids):
            if pid in gt:
                found_rank = i + 1
                break

        n += 1
        if found_rank is not None:
            if found_rank == 1:
                recall_at_1 += 1
            if found_rank <= 5:
                recall_at_5 += 1
            mrr_sum += 1.0 / found_rank

    if n == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0, "time": total_time}
    return {
        "recall@1": recall_at_1 / n,
        "recall@5": recall_at_5 / n,
        "mrr": mrr_sum / n,
        "time": total_time
    }

def grid_search(queries: List[Dict[str,Any]], steps: int, k: int, rerank_k: int):
    results = []
    for i in range(steps):
        alpha = i / (steps - 1) if steps > 1 else 1.0
        beta = 1.0 - alpha
        print(f"Evaluando alpha={alpha:.2f} beta={beta:.2f} (rerank_k={rerank_k}) ...")
        stats = eval_queries(queries, alpha, beta, k=k, rerank_k=rerank_k)
        print(f" -> recall@5: {stats['recall@5']:.3f} recall@1: {stats['recall@1']:.3f} mrr: {stats['mrr']:.3f} time: {stats['time']:.1f}s")
        results.append({
            "alpha": alpha, "beta": beta, "rerank_k": rerank_k, **stats
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Fichero JSON con queries (gt_ids o pid).")
    parser.add_argument("--steps", type=int, default=11, help="Número de pasos en grid alpha (default 11).")
    parser.add_argument("--k", type=int, default=6, help="k resultados a solicitar al API (default 6).")
    parser.add_argument("--rerank_k", type=int, default=20, help="k para re-ranker (default 20).")
    args = parser.parse_args()

    queries = load_queries(args.eval)
    print(f"Queries a evaluar: {len(queries)}")
    res = grid_search(queries, steps=args.steps, k=args.k, rerank_k=args.rerank_k)
    res_sorted = sorted(res, key=lambda x: x["mrr"], reverse=True)

    print("\nTop combos (por mrr):")
    for r in res_sorted[:6]:
        print(f"alpha={r['alpha']:.2f} beta={r['beta']:.2f} rerank_k={r['rerank_k']} recall@5={r['recall@5']:.3f} recall@1={r['recall@1']:.3f} mrr={r['mrr']:.3f}")

    out_path = "backend/eval_results_rerank.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": res, "sorted": res_sorted}, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {out_path}")

if __name__ == "__main__":
    main()
