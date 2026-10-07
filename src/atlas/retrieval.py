"""Batched dense retrieval with a content-addressed embedding cache."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .chunking import Chunk


class Retriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 *, model=None, cache_dir=None, batch_size=32):
        if model is None:
            from .models import load_embedding_model
            model = load_embedding_model(model_name)
        self.model = model
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size
        self.chunks = []
        self.embeddings = None

    def index(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No readable chunks were found.")
        chunks = list(chunks)
        texts = [piece.text for piece in chunks]
        model_config = (self.model.get_config_dict()
                        if hasattr(self.model, "get_config_dict") else {})
        transformer = self.model[0] if hasattr(self.model, "__getitem__") else None
        revision = getattr(getattr(getattr(transformer, "auto_model", None),
                                   "config", None), "_commit_hash", None)
        key = hashlib.sha256(json.dumps(
            ["atlas-embeddings-v1", self.model_name, revision, model_config, texts],
            ensure_ascii=False, sort_keys=True, default=str
        ).encode()).hexdigest()
        path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        vectors = None
        if path is not None and path.exists():
            try:
                vectors = np.load(path, allow_pickle=False)
                self._check_vectors(vectors, len(chunks))
            except (ValueError, TypeError, OSError, EOFError):
                vectors = None
        if vectors is None:
            vectors = np.asarray(self.model.encode_document(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False
            ), dtype=np.float32)
            self._check_vectors(vectors, len(chunks))
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy",
                                                  delete=False) as file:
                    temporary = Path(file.name)
                    np.save(file, vectors, allow_pickle=False)
                try:
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        self.chunks, self.embeddings = chunks, vectors

    @staticmethod
    def _check_vectors(vectors, count):
        if (vectors.ndim != 2 or vectors.shape[0] != count
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.01)):
            raise ValueError("Invalid or unnormalized embedding matrix.")

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if self.embeddings is None:
            raise RuntimeError("Index chunks before you search.")
        if not query.strip() or k <= 0:
            raise ValueError("Provide a nonempty query and positive k.")
        query_vector = np.asarray(self.model.encode_query(
            query, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False
        ), dtype=np.float32)
        if (query_vector.shape != (self.embeddings.shape[1],)
                or not np.isfinite(query_vector).all()):
            raise ValueError("Invalid query embedding.")
        scores = self.embeddings @ query_vector
        best = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]
