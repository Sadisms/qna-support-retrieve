import threading

from sentence_transformers import SentenceTransformer


class Embedder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def encode(self, texts, convert_to_numpy=True):
        embeddings = self.model.encode(texts, convert_to_numpy=convert_to_numpy, normalize_embeddings=True)
        return embeddings

