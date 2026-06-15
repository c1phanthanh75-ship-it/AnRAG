from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from anrag.embedding import EmbeddingBackend
from anrag.models import Chunk, SearchHit
from anrag.text import simple_tokens


class DenseIndex:
    def __init__(self, path: str | Path, embedder: EmbeddingBackend, namespace: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.namespace = namespace
        self.index_path = self.path / f"{namespace}.faiss"
        self.ids_path = self.path / f"{namespace}.ids.json"
        self.index: faiss.Index | None = None
        self.ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.ids = [chunk.id for chunk in chunks]
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        dim = vectors.shape[1] if vectors.size else self.embedder.dim
        self.index = faiss.IndexFlatIP(dim)
        if len(vectors):
            self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_path))
        self.ids_path.write_text(json.dumps(self.ids, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.index_path.exists() or not self.ids_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        return True

    def search(self, query: str, top_k: int = 8, valid_ids: set[str] | None = None) -> list[SearchHit]:
        if self.index is None:
            if not self.load():
                return []
        if not self.ids or self.index is None:
            return []
        vector = self.embedder.encode([query])
        search_k = len(self.ids) if valid_ids is not None else min(top_k, len(self.ids))
        scores, positions = self.index.search(vector, search_k)
        hits: list[SearchHit] = []
        for score, pos in zip(scores[0], positions[0], strict=False):
            if pos < 0:
                continue
            chunk_id = self.ids[pos]
            if valid_ids is not None and chunk_id not in valid_ids:
                continue
            hits.append(SearchHit(chunk_id=chunk_id, score=float(score), source=self.namespace))
            if valid_ids is not None and len(hits) >= top_k:
                break
        return hits


class SparseIndex:
    def __init__(self, chunks: list[Chunk], namespace: str):
        self.chunks = chunks
        self.namespace = namespace
        self.ids = [chunk.id for chunk in chunks]
        self.bm25 = BM25Okapi([simple_tokens(chunk.text) for chunk in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(simple_tokens(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(chunk_id=self.ids[index], score=float(scores[index]), source=f"{self.namespace}:bm25")
            for index in order
            if scores[index] > 0
        ]
