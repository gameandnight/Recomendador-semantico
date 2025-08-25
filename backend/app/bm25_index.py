import os, pickle
from rank_bm25 import BM25Okapi

BM25_TOKENS_PATH = os.environ.get("BM25_TOKENS_PATH", "backend/data/corpus_tokens.pkl")

class BM25Index:
    def __init__(self, tokens_path=BM25_TOKENS_PATH):
        self.tokens_path = tokens_path
        self.bm25 = None
        self.corpus_tokens = []
        self._ensure_loaded()

    def _ensure_loaded(self):
        if not self.bm25:
            if not os.path.exists(self.tokens_path):
                # no hay corpus tokenizado preparado
                self.corpus_tokens = []
                self.bm25 = None
                return
            with open(self.tokens_path, "rb") as f:
                self.corpus_tokens = pickle.load(f)
            if self.corpus_tokens:
                self.bm25 = BM25Okapi(self.corpus_tokens)
            else:
                self.bm25 = None

    def get_scores(self, query):
        """
        Devuelve un array (list) de scores ordenados por índice del documento.
        Si bm25 no está disponible retorna [].
        """
        self._ensure_loaded()
        if not self.bm25:
            return []
        qtokens = query.lower().split()
        return self.bm25.get_scores(qtokens)

    def get_top_n(self, query, n=10):
        """
        Devuelve los índices de los top-n documentos según BM25.
        """
        self._ensure_loaded()
        if not self.bm25:
            return []
        qtokens = query.lower().split()
        top_n = self.bm25.get_top_n(qtokens, range(len(self.corpus_tokens)), n=n)
        # note: rank_bm25.get_top_n returns the actual documents; our documents are indices, so adapt:
        # Instead, compute scores and sort:
        scores = list(self.bm25.get_scores(qtokens))
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s in indexed[:n]]
