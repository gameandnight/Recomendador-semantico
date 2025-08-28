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
  };

  const formatScore = (s) => {
    return (Number(s) * 100).toFixed(2) + "%";
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter") doSearch();
  };

  // --- NUEVA LÓGICA PARA IMÁGENES ---
  const getImageUrl = (product, idx) => {
    // 1) si hay URL absoluta en product.image, usarla
    const img = product?.image;
    if (img && /^https?:\/\//i.test(img)) {
      return img;
    }

    // 2) intentar keyword por categoría o título (palabra significativa)
    const title = (product?.title || "").toString().trim();
    const category = (product?.category || "").toString().trim();

    const pickKeywordFromText = (text) => {
      if (!text) return "";
      // split y busca primera palabra larga (>2), ignora signos
      const toks = text.split(/[\s,._\-\/]+/).map(t => t.replace(/[^\wáéíóúñüÁÉÍÓÚÑÜ]+/g, ""));
      for (let t of toks) {
        if (t && t.length > 2) return t.toLowerCase();
      }
      return "";
    };

    const keyword = pickKeywordFromText(category) || pickKeywordFromText(title);

    // 3) si tenemos keyword -> usar Unsplash Source por keyword (+ sig para variar)
    if (keyword) {
      // tamaño 400x400; sig ayuda a reducir repeticiones entre tarjetas
      const sig = encodeURIComponent(product?.id ?? idx);
      return `https://source.unsplash.com/400x400/?${encodeURIComponent(keyword)}&sig=${sig}`;
    }

    // 4) fallback determinista con Picsum (seed por id o índice)
    const seed = encodeURIComponent(product?.id ?? idx);
    return `https://picsum.photos/seed/${seed}/400/400`;
  };
  // --- fin de la nueva lógica ---

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1 className="title">Supermercado</h1>
          <p className="subtitle">Recomendador Semántico — demo</p>
        </div>
      </header>

      <section className="search-area">
        <div className="search-row">
          <input
            className="search-input"
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Busca por ejemplo: 'Patatas'"
            aria-label="Buscar productos"
            autoFocus
          />
          <button className="btn primary" onClick={doSearch} disabled={!q || loading}>
            {loading ? <span className="spinner" aria-hidden /> : "Buscar"}
          </button>
        </div>

        <div className="controls-row" role="group" aria-label="Ajustes de búsqueda">
          <div className="control">
            <label>k</label>
            <input type="number" value={k} onChange={e => setK(e.target.value)} />
          </div>

          <div className="control">
            <label>Embeddings α</label>
            <input step="0.05" min="0" max="1" type="number" value={alpha} onChange={e => setAlpha(e.target.value)} />
          </div>

          <div className="control">
            <label>BM25 β</label>
            <input step="0.05" min="0" max="1" type="number" value={beta} onChange={e => setBeta(e.target.value)} />
          </div>

          <div className="control checkbox">
            <label>Use Re-ranker</label>
            <input type="checkbox" checked={useReranker} onChange={e => setUseReranker(e.target.checked)} />
          </div>

          <div className="control">
            <label>Rerank K</label>
            <input type="number" value={rerankK} onChange={e => setRerankK(e.target.value)} />
          </div>

          
        </div>
      </section>

      <section className="results-area">
        {results.length === 0 && !loading && <div className="empty">No hay resultados aún.</div>}

        <div className="results-grid">
          {results.map((r, i) => (
            <article key={r.product.id || i} className="product-card">
              <div className="product-media">
                {r.product.image ? (
                  // si viene imagen (validada por getImageUrl si tiene http), la usamos
                  <img src={getImageUrl(r.product, i)} alt={r.product.title} />
                ) : (
                  // si no viene, getImageUrl devolverá Unsplash o Picsum
                  <img src={getImageUrl(r.product, i)} alt={r.product.title} />
                )}
              </div>

              <div className="product-body">
                <h3 className="product-title">{r.product.title}</h3>
                <p className="product-desc">{r.product.description}</p>

                <div className="product-meta">
                  <div className="price">€{r.product.price}</div>

                  <div className="score">
                    <div className="score-bar" style={{ width: `${Math.min(100, Number(r.score) * 100)}%` }} />
                    <div className="score-text">{formatScore(r.score)}</div>
                  </div>
                </div>

                <div className="product-footer">
                  <span className={`badge ${r.source === "semantic" ? "semantic" : "textual"}`}>
                    {r.source === "semantic" ? "Semántico (Embeddings)" : "Textual (BM25)"}
                  </span>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>

      <footer className="footer">
        <small>Versión demo · Datos de ejemplo · Proyecto para portfolio</small>
      </footer>
    </div>
  );
}

