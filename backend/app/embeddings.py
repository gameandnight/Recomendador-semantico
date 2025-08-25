from sentence_transformers import SentenceTransformer
import os
import numpy as np

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        texts: list[str]
        returns: np.ndarray normalized (len(texts), dim)
        """
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalizar (para usar IndexFlatIP y tratar inner product como cosine)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        return emb / norm
