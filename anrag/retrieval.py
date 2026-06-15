from __future__ import annotations

import time
from collections import OrderedDict

from anrag.embedding import EmbeddingBackend
from anrag.index import DenseIndex, SparseIndex
from anrag.llm import OllamaLLM
from anrag.models import Chunk, RetrievalResult, SearchHit
from anrag.rerank import CrossEncoderReranker
from anrag.store import SQLiteTreeStore
from anrag.text import normalize_query, token_count


def looks_ambiguous(query: str) -> bool:
    words = query.split()
    vague = {"it", "this", "that", "method", "algorithm", "cái này", "phương pháp", "thuật toán"}
    return len(words) <= 5 or query.lower() in vague


def reciprocal_rank_fusion(
    *hit_lists: list[SearchHit],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchHit]:
    """Weighted Reciprocal Rank Fusion across multiple ranked lists."""
    if weights is None:
        weights = [1.0] * len(hit_lists)
    if len(weights) != len(hit_lists):
        raise ValueError("weights length must match number of hit lists")

    rrf_scores: dict[str, float] = {}
    best_source: dict[str, str] = {}
    for hit_list, weight in zip(hit_lists, weights, strict=False):
        for rank, hit in enumerate(hit_list, start=1):
            contribution = weight / (k + rank)
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + contribution
            if hit.chunk_id not in best_source or hit.score > rrf_scores.get(hit.chunk_id, 0.0):
                best_source[hit.chunk_id] = hit.source

    return [
        SearchHit(chunk_id=cid, score=score, source=best_source.get(cid, "rrf"))
        for cid, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    ]


class AnchorRetriever:
    def __init__(
        self,
        store: SQLiteTreeStore,
        index_dir: str,
        embedder: EmbeddingBackend,
        llm: OllamaLLM | None = None,
        reranker: CrossEncoderReranker | None = None,
        confidence_threshold: float = 0.32,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker
        self.confidence_threshold = confidence_threshold
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.anchor_index = DenseIndex(index_dir, embedder, "anchors")
        self.chunk_index = DenseIndex(index_dir, embedder, "chunks")

    def rebuild_indexes(self) -> None:
        self.anchor_index.build(self.store.anchor_chunks())
        self.chunk_index.build(self.store.all_chunks())

    def _prepare_query(self, query: str, *, rewrite_query: bool) -> tuple[str, str]:
        q = normalize_query(query)
        rewritten = q
        if rewrite_query and self.llm and looks_ambiguous(q):
            try:
                rewritten = self.llm.rewrite_query(q)
            except Exception:
                rewritten = q
        return q, rewritten

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        *,
        dense_index: DenseIndex,
        chunk_pool: list[Chunk],
        namespace: str,
        valid_ids: set[str] | None = None,
    ) -> tuple[list[SearchHit], str]:
        fetch_k = max(top_k * 3, 20)

        dense_hits = dense_index.search(query, top_k=fetch_k, valid_ids=valid_ids)

        if valid_ids:
            pool = [chunk for chunk in chunk_pool if chunk.id in valid_ids]
        else:
            pool = chunk_pool

        sparse_hits = SparseIndex(pool, namespace).search(query, top_k=fetch_k)

        if not dense_hits and not sparse_hits:
            return [], "empty"
        if not sparse_hits:
            return dense_hits[:top_k], "dense_only"

        fused = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            k=self.rrf_k,
            weights=[self.dense_weight, self.sparse_weight],
        )
        return fused[:top_k], "rrf"

    def _hybrid_anchor_search(
        self,
        query: str,
        top_k: int,
        valid_ids: set[str] | None = None,
    ) -> tuple[list[SearchHit], str]:
        pool = (
            [chunk for chunk in self.store.anchor_chunks() if chunk.id in valid_ids]
            if valid_ids
            else self.store.anchor_chunks()
        )
        return self._hybrid_search(
            query,
            top_k,
            dense_index=self.anchor_index,
            chunk_pool=pool,
            namespace="anchors",
            valid_ids=valid_ids,
        )

    def _hybrid_chunk_search(
        self,
        query: str,
        top_k: int,
        valid_ids: set[str] | None = None,
    ) -> tuple[list[SearchHit], str]:
        pool = (
            [chunk for chunk in self.store.all_chunks() if chunk.id in valid_ids]
            if valid_ids
            else self.store.all_chunks()
        )
        return self._hybrid_search(
            query,
            top_k,
            dense_index=self.chunk_index,
            chunk_pool=pool,
            namespace="chunks",
            valid_ids=valid_ids,
        )

    def retrieve(
        self,
        query: str,
        budget_tokens: int = 1200,
        top_k: int = 5,
        top_p: float = 0.85,
        generate_answer: bool = True,
        stream_answer: bool = False,
        doc_ids: list[str] | None = None,
        rewrite_query: bool = True,
    ) -> RetrievalResult:
        start_time = time.time()
        q, rewritten = self._prepare_query(query, rewrite_query=rewrite_query)

        valid_ids: set[str] | None = None
        if doc_ids:
            valid_ids = set()
            for did in doc_ids:
                valid_ids.update(chunk.id for chunk in self.store.anchor_chunks(did))

        fused_hits, fusion_mode = self._hybrid_anchor_search(rewritten, top_k=top_k, valid_ids=valid_ids)
        anchors = self._top_p(fused_hits, top_p)
        confidence = max((hit.score for hit in anchors), default=0.0)

        candidates = self.expand(anchors)
        if self.reranker:
            candidates = self.reranker.rerank(rewritten, candidates)
        contexts = self.budget_select(candidates, budget_tokens)
        answer = self._maybe_generate_answer(query, contexts, generate_answer, stream_answer)

        latency = time.time() - start_time
        return RetrievalResult(
            answer=answer,
            contexts=contexts,
            anchors=anchors,
            trace={
                "normalized_query": q,
                "rewritten_query": rewritten,
                "confidence": confidence,
                "fusion_mode": fusion_mode,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
                "candidate_count": len(candidates),
                "budget_tokens": budget_tokens,
                "mode": "anchor_rag",
                "rewrite_query": rewrite_query,
                "latency_seconds": round(latency, 3),
            },
        )

    def anchor_only(
        self,
        query: str,
        budget_tokens: int = 1200,
        top_k: int = 5,
        top_p: float = 0.85,
        generate_answer: bool = True,
        stream_answer: bool = False,
        doc_ids: list[str] | None = None,
        rewrite_query: bool = True,
    ) -> RetrievalResult:
        """Anchor retrieval without tree expansion — isolates anchor contribution."""
        start_time = time.time()
        q, rewritten = self._prepare_query(query, rewrite_query=rewrite_query)

        valid_ids: set[str] | None = None
        if doc_ids:
            valid_ids = set()
            for did in doc_ids:
                valid_ids.update(chunk.id for chunk in self.store.anchor_chunks(did))

        fused_hits, fusion_mode = self._hybrid_anchor_search(rewritten, top_k=top_k, valid_ids=valid_ids)
        anchors = self._top_p(fused_hits, top_p)
        confidence = max((hit.score for hit in anchors), default=0.0)

        seen = set()
        unique_chunk_ids = []
        for hit in anchors:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                unique_chunk_ids.append(hit.chunk_id)
        candidates = self.store.get_chunks(unique_chunk_ids)
        if self.reranker:
            candidates = self.reranker.rerank(rewritten, candidates)
        contexts = self.budget_select(candidates, budget_tokens)
        answer = self._maybe_generate_answer(query, contexts, generate_answer, stream_answer)

        latency = time.time() - start_time
        return RetrievalResult(
            answer=answer,
            contexts=contexts,
            anchors=anchors,
            trace={
                "normalized_query": q,
                "rewritten_query": rewritten,
                "confidence": confidence,
                "fusion_mode": fusion_mode,
                "candidate_count": len(candidates),
                "budget_tokens": budget_tokens,
                "mode": "anchor_only",
                "rewrite_query": rewrite_query,
                "latency_seconds": round(latency, 3),
            },
        )

    def plain_rag(
        self,
        query: str,
        budget_tokens: int = 1200,
        top_k: int = 20,
        generate_answer: bool = True,
        stream_answer: bool = False,
        doc_ids: list[str] | None = None,
        rewrite_query: bool = True,
    ) -> RetrievalResult:
        """Plain RAG: hybrid dense+sparse over all chunks — no anchor search or expansion."""
        start_time = time.time()
        q, rewritten = self._prepare_query(query, rewrite_query=rewrite_query)

        valid_ids: set[str] | None = None
        if doc_ids:
            valid_ids = set()
            for did in doc_ids:
                valid_ids.update(chunk.id for chunk in self.store.all_chunks(did))

        hits, fusion_mode = self._hybrid_chunk_search(rewritten, top_k=top_k, valid_ids=valid_ids)
        seen = set()
        unique_chunk_ids = []
        for hit in hits:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                unique_chunk_ids.append(hit.chunk_id)
        candidates = self.store.get_chunks(unique_chunk_ids)
        if self.reranker:
            candidates = self.reranker.rerank(rewritten, candidates)
        contexts = self.budget_select(candidates, budget_tokens)
        answer = self._maybe_generate_answer(query, contexts, generate_answer, stream_answer)

        latency = time.time() - start_time
        return RetrievalResult(
            answer=answer,
            contexts=contexts,
            anchors=hits,
            trace={
                "normalized_query": q,
                "rewritten_query": rewritten,
                "fusion_mode": fusion_mode,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
                "candidate_count": len(candidates),
                "budget_tokens": budget_tokens,
                "mode": "plain_rag",
                "rewrite_query": rewrite_query,
                "latency_seconds": round(latency, 3),
            },
        )

    def baseline(
        self,
        query: str,
        budget_tokens: int = 1200,
        top_k: int = 8,
        generate_answer: bool = True,
        stream_answer: bool = False,
        doc_ids: list[str] | None = None,
        rewrite_query: bool = True,
    ) -> RetrievalResult:
        """Backward-compatible alias for :meth:`plain_rag`."""
        return self.plain_rag(
            query,
            budget_tokens=budget_tokens,
            top_k=top_k,
            generate_answer=generate_answer,
            stream_answer=stream_answer,
            doc_ids=doc_ids,
            rewrite_query=rewrite_query,
        )

    def expand(self, anchors: list[SearchHit]) -> list[Chunk]:
        ordered: OrderedDict[str, Chunk] = OrderedDict()
        for hit in anchors:
            anchor = self.store.get_chunk(hit.chunk_id)
            if not anchor:
                continue
            self._add(ordered, anchor)
            parent = self.store.parent(anchor)
            if parent:
                self._add(ordered, parent)
            for sibling in self.store.siblings(anchor)[:4]:
                self._add(ordered, sibling)
            for neighbor in self.store.local_neighbors(anchor, radius=2):
                self._add(ordered, neighbor)
        return list(ordered.values())

    def budget_select(self, candidates: list[Chunk], budget_tokens: int) -> list[Chunk]:
        if not candidates:
            return []
        selected: list[Chunk] = []
        seen_texts: set[str] = set()
        used = 0
        for chunk in candidates:
            # Normalize text (lowercase, strip whitespace) to identify duplicates
            norm_text = " ".join(chunk.text.lower().split())
            if norm_text in seen_texts:
                continue
            cost = chunk.token_count or token_count(chunk.text)
            if used + cost <= budget_tokens:
                selected.append(chunk)
                seen_texts.add(norm_text)
                used += cost
        return selected

    def _maybe_generate_answer(
        self,
        query: str,
        contexts: list[Chunk],
        generate_answer: bool,
        stream_answer: bool,
    ) -> str | object:
        if not generate_answer or not self.llm:
            return ""
        try:
            if stream_answer:
                return self.llm.answer_stream(query, contexts)
            return self.llm.answer(query, contexts)
        except Exception as exc:
            return f"LLM generation failed: {exc}"

    @staticmethod
    def _add(ordered: OrderedDict[str, Chunk], chunk: Chunk) -> None:
        if chunk.id not in ordered:
            ordered[chunk.id] = chunk

    @staticmethod
    def _merge_hits(*groups: list[SearchHit]) -> list[SearchHit]:
        merged: dict[str, SearchHit] = {}
        for group in groups:
            for hit in group:
                current = merged.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    merged[hit.chunk_id] = hit
        return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)

    @staticmethod
    def _top_p(hits: list[SearchHit], top_p: float) -> list[SearchHit]:
        if not hits:
            return []
        positive = [max(0.0, hit.score) for hit in hits]
        total = sum(positive)
        if total <= 0:
            return hits[: min(3, len(hits))]
        selected: list[SearchHit] = []
        running = 0.0
        for hit, score in zip(hits, positive, strict=False):
            selected.append(hit)
            running += score / total
            if running >= top_p:
                break
        return selected
