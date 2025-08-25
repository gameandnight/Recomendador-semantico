#!/usr/bin/env python3
"""
evaluate_with_evalset.py
Evaluación de un evalset (queries con gt_ids o pid).
Salida: imprime métricas por combinación alpha/beta y guarda backend/eval_results.json
Uso:
    python backend/evaluate_with_evalset.py --eval backend/eval_queries_with_gtids.json --steps 11 --k 6
"""

import argparse
import json
import time
import requests
import sys
import math
from typing import List, Dict, Any

API = "http://localhost:8000/search"
TIMEOUT = 10  # seconds for requests

def load_queries(path: str) -> List[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for q in raw:
        # normalizar: aceptar gt_ids (lista) o pid (string)
        entry = {"q": q.get("q") or q.get("query") or q.get("title") or ""}
        if "gt_ids" in q and isinstance(q["gt_ids"], list):
            entry["gt_ids"] = q["gt_ids"]
        elif "pid" in q and q["pid"]:
            entry["gt_ids"] = [q["pid"]]
        else:
            # intentar usar 'id' u 'ids'
            if "id" in q:
                entry["gt_ids"] = [q["id"]]
            elif "ids" in q and isinstance(q["ids"], list):
                entry["gt_ids"] = q["ids"]
            else:
                entry["gt_ids"] = []
        out.append(entry)
    return out

def call_search(q: str, k: int, alpha: float, beta: float) -> Dict[str,Any]:
    payload = {"q": q, "k": k, "alpha": alpha, "beta": beta, "use_reranker": False}
    try:
        r = requests.post(API, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Warning: search request failed for q='{q[:80]}' -> {e}", file=sys.stderr)
        return {"query": q, "results": []}

def eval_queries(queries: List[Dict[str,Any]], alpha: float, beta: float, k: int):
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
            # no ground truth -> ignórala
            continue
        start = time.time()
        resp = call_search(q["q"], k=k, alpha=alpha, beta=beta)
        elapsed = time.time() - start
        total_time += elapsed

        results = resp.get("results", [])
        ids = []
        # cada resultado puede tener estructura product.id
        for r in results:
            p = r.get("product", {})
            pid = p.get("id") or p.get("pid") or p.get("product_id")
            if pid:
                ids.append(str(pid))

        # métricas
        found_rank = None
        for i, pid in enumerate(ids):
            if pid in gt:
                found_rank = i + 1
                break

        if found_rank is not None:
            n += 1
            # recall@1
            if found_rank == 1:
                recall_at_1 += 1
            # recall@5
            if found_rank <= 5:
                recall_at_5 += 1
            # mrr
            mrr_sum += 1.0 / found_rank
        else:
            # si no se encontró, cuenta en n igual (para denominador)
            n += 1

    if n == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0, "time": total_time}
    return {
        "recall@1": recall_at_1 / n,
        "recall@5": recall_at_5 / n,
        "mrr": mrr_sum / n,
        "time": total_time
    }

def grid_search(queries: List[Dict[str,Any]], steps: int, k: int):
    results = []
    for i in range(steps):
        alpha = i / (steps - 1) if steps > 1 else 1.0
        beta = 1.0 - alpha
        print(f"Evaluando alpha={alpha:.2f} beta={beta:.2f} ...")
        stats = eval_queries(queries, alpha, beta, k=k)
        print(f" -> recall@5: {stats['recall@5']:.3f} recall@1: {stats['recall@1']:.3f} mrr: {stats['mrr']:.3f} time: {stats['time']:.1f}s")
        results.append({
            "alpha": alpha, "beta": beta, **stats
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Fichero JSON con queries (gt_ids o pid).")
    parser.add_argument("--steps", type=int, default=11, help="Número de pasos en grid alpha (default 11).")
    parser.add_argument("--k", type=int, default=6, help="k resultados a solicitar al API (default 6).")
    args = parser.parse_args()

    queries = load_queries(args.eval)
    print(f"Queries a evaluar: {len(queries)}")
    res = grid_search(queries, steps=args.steps, k=args.k)
    # ordenar por mrr
    res_sorted = sorted(res, key=lambda x: x["mrr"], reverse=True)
    print("\nTop combos (por mrr):")
    for r in res_sorted[:6]:
        print(f"alpha={r['alpha']:.2f} beta={r['beta']:.2f} recall@5={r['recall@5']:.3f} recall@1={r['recall@1']:.3f} mrr={r['mrr']:.3f}")

    out_path = "backend/eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": res, "sorted": res_sorted}, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {out_path}")

if __name__ == "__main__":
    main()
