const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function search(query, k = 6, alpha = 0.7, beta = 0.3, use_reranker = false, rerank_k = 50) {
    const res = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: query, k, alpha, beta, use_reranker, rerank_k })
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Search failed");
    }
    return await res.json();
}

