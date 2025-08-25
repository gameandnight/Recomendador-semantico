import { useState } from "react";
import { search } from "./api";

export default function App() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [alpha, setAlpha] = useState(0.7);
  const [beta, setBeta] = useState(0.3);
  const [k, setK] = useState(6);
  const [useReranker, setUseReranker] = useState(false);
  const [rerankK, setRerankK] = useState(50);

  const doSearch = async () => {
    if (!q) return;
    setLoading(true);
    try {
      const data = await search(q, Number(k), Number(alpha), Number(beta), useReranker, Number(rerankK));
      setResults(data.results || []);
    } catch (e) {
      alert(e.message);
    } finally { setLoading(false); }
  }

  const formatScore = (s) => {
    return (Number(s) * 100).toFixed(2) + "%";
  }

  return (
    <div className="container" style={{ maxWidth: 900, margin: "20px auto", fontFamily: "Arial, sans-serif" }}>
      <h1>Supermercado (Recomendador Semántico)</h1>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input style={{ flex: 1, padding: 8 }} value={q} onChange={e => setQ(e.target.value)} placeholder="Busca por ejemplo: 'chaqueta impermeable'" />
        <button onClick={doSearch} disabled={!q || loading}>{loading ? "Buscando..." : "Buscar"}</button>
      </div>

      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <div>
          <label>k</label><br />
          <input type="number" value={k} onChange={e => setK(e.target.value)} style={{ width: 60 }} />
        </div>
        <div>
          <label>Embeddings α</label><br />
          <input step="0.05" min="0" max="1" type="number" value={alpha} onChange={e => setAlpha(e.target.value)} style={{ width: 80 }} />
        </div>
        <div>
          <label>BM25 β</label><br />
          <input step="0.05" min="0" max="1" type="number" value={beta} onChange={e => setBeta(e.target.value)} style={{ width: 80 }} />
        </div>
        <div>
          <label>Use Re-ranker</label><br />
          <input type="checkbox" checked={useReranker} onChange={e => setUseReranker(e.target.checked)} />
        </div>
        <div>
          <label>Rerank K</label><br />
          <input type="number" value={rerankK} onChange={e => setRerankK(e.target.value)} style={{ width: 80 }} />
        </div>
        <div style={{ alignSelf: "end" }}>
          <button onClick={doSearch} disabled={!q || loading}>Buscar (con α/β)</button>
        </div>
      </div>

      <div>
        {results.length === 0 && <p>No hay resultados aún.</p>}
        {results.map((r, i) => (
          <div key={r.product.id || i} style={{ border: "1px solid #ddd", padding: 12, borderRadius: 6, marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>{r.product.title}</h3>
            <p style={{ margin: "6px 0" }}>{r.product.description}</p>
            <p style={{ margin: "6px 0" }}><strong>Precio:</strong> €{r.product.price}</p>
            <small>Score: {formatScore(r.score)}</small>
            <div style={{ marginTop: 6 }}>
              <em>Origen: </em>
              <span style={{
                padding: "4px 8px",
                borderRadius: 6,
                background: r.source === "semantic" ? "#e0f7fa" : "#fff3e0",
                fontSize: 12
              }}>
                {r.source === "semantic" ? "Semántico (Embeddings)" : "Textual (BM25)"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
