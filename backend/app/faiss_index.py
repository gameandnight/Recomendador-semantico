import faiss
import numpy as np
import os
import pickle

INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "backend/data/faiss.index")
META_PATH = os.environ.get("FAISS_META_PATH", "backend/data/meta.pkl")


class FaissIndex:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product (usamos vectores normalizados)
        self.meta = []

    def add(self, vectors: np.ndarray, metas: list):
        if vectors.dtype != np.float32:
            vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.meta.extend(metas)

    

    def save(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.meta, f)

    def load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.meta = pickle.load(f)
            return True
        return False
    
    def search(self, vector: np.ndarray, k=5):
        """
        Busca los k vecinos más cercanos para cada vector en `vector`.
        Devuelve una lista de filas; cada fila es una lista de {"score", "meta"}.
        Esta versión es defensiva: ignora índices inválidos que pudieran producirse.
        """
        # Si no hay vectores en el índice, devolvemos listas vacías por cada query
        if self.index.ntotal == 0:
            return [[] for _ in range(vector.shape[0])]

        # Aseguramos que k no exceda el número de vectores en el índice
        k_eff = min(int(k), max(1, self.index.ntotal))

        # Ejecuta búsqueda
        if vector.dtype != np.float32:
            vector = vector.astype('float32')
        D, I = self.index.search(vector, k_eff)

        results = []
        for distances, indices in zip(D, I):
            row = []
            for dist, idx in zip(distances, indices):
                # IGNORAR índices inválidos (faiss puede devolver -1 si falta vecino)
                if idx is None or int(idx) < 0:
                    continue
                # Si meta está fuera de rango, también se ignora (evita IndexError)
                if idx >= len(self.meta):
                    # opcional: registrar advertencia
                    # print(f"Warning: faiss returned idx {idx} but meta length is {len(self.meta)}")
                    continue
                item_meta = self.meta[int(idx)]
                row.append({"score": float(dist), "meta": item_meta})
            results.append(row)
        return results

