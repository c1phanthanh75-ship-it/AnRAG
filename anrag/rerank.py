from __future__ import annotations

import math
from typing import Literal

import numpy as np

from anrag.models import Chunk

# Task types used to select the score threshold before passing to the LLM.
# "factual"  → precise retrieval needed, higher threshold
# "summary"  → broader context okay, lower threshold
# "default"  → balanced middle ground
TaskType = Literal["factual", "summary", "default"]

_TASK_THRESHOLDS: dict[TaskType, float] = {
    "factual": 0.35,
    "summary": 0.10,
    "default": 0.20,
}


def _sigmoid_calibrate(raw_scores: list[float]) -> list[float]:
    """Map cross-encoder raw logits to [0, 1] via sigmoid.

    CrossEncoder models trained with MSE often output scores in a wide
    range (e.g. -5 … +5). A simple sort is fine for ranking, but the
    raw numbers are hard to threshold across models.  Sigmoid squashes
    them into a stable probability-like range so task thresholds are
    model-agnostic.
    """
    return [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]


def _mmr_select(
    chunks: list[Chunk],
    scores: list[float],
    embeddings: np.ndarray | None,
    top_n: int,
    lambda_mmr: float,
) -> list[Chunk]:
    """Maximal Marginal Relevance selection.

    Balances relevance (score) against redundancy (cosine similarity to
    already-selected items).  When embeddings are unavailable the method
    falls back to pure-score ordering.

    lambda_mmr=1.0  → pure relevance (equivalent to no MMR)
    lambda_mmr=0.0  → maximum diversity
    lambda_mmr=0.5  → balanced (recommended default)
    """
    if embeddings is None or len(embeddings) == 0:
        return chunks[:top_n]

    n = len(chunks)
    remaining = list(range(n))
    selected_indices: list[int] = []

    while remaining and len(selected_indices) < top_n:
        if not selected_indices:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: scores[i])
        else:
            selected_embs = embeddings[selected_indices]  # (k, d)

            def mmr_score(i: int) -> float:
                rel = scores[i]
                emb = embeddings[i]  # (d,)
                sims = selected_embs @ emb  # (k,)
                max_sim = float(np.max(sims)) if len(sims) else 0.0
                return lambda_mmr * rel - (1.0 - lambda_mmr) * max_sim

            best = max(remaining, key=mmr_score)

        selected_indices.append(best)
        remaining.remove(best)

    return [chunks[i] for i in selected_indices]


class CrossEncoderReranker:
    """Cross-encoder reranker with sigmoid score calibration and MMR diversity.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g.
        ``"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"``.
    lambda_mmr:
        MMR diversity trade-off in [0, 1].  0.5 is balanced; 1.0 disables MMR.
    mmr_top_n:
        Maximum candidates to keep after MMR (before budget trimming in
        AnchorRetriever).  ``None`` keeps all calibrated survivors.
    """

    def __init__(
        self,
        model_name: str,
        lambda_mmr: float = 0.5,
        mmr_top_n: int | None = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.lambda_mmr = lambda_mmr
        self.mmr_top_n = mmr_top_n
        # Try to load the paired bi-encoder for MMR similarity computation.
        # If unavailable, MMR falls back to pure relevance.
        self._bi_encoder = None
        try:
            from sentence_transformers import SentenceTransformer

            self._bi_encoder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:
            pass

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        task_type: TaskType = "default",
        return_scores: bool = False,
    ) -> list[Chunk]:
        """Rerank chunks and optionally filter by calibrated score threshold.

        Parameters
        ----------
        query:
            The user query (or rewritten query).
        chunks:
            Candidate chunks from the expansion stage.
        task_type:
            Controls the minimum calibrated score below which chunks are
            dropped before MMR.  Use ``"factual"`` for precise QA,
            ``"summary"`` for broader synthesis, ``"default"`` otherwise.
        return_scores:
            If ``True``, stores calibrated scores in each chunk's metadata
            under the key ``"rerank_score"`` for downstream inspection.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = [float(s) for s in self.model.predict(pairs)]
        calibrated = _sigmoid_calibrate(raw_scores)

        # Apply task-aware threshold
        threshold = _TASK_THRESHOLDS.get(task_type, _TASK_THRESHOLDS["default"])
        paired = [(chunk, score) for chunk, score in zip(chunks, calibrated, strict=False) if score >= threshold]

        if not paired:
            # If all chunks fall below threshold, keep the top-3 by score
            # rather than returning nothing — graceful degradation.
            paired = sorted(
                zip(chunks, calibrated, strict=False),
                key=lambda item: item[1],
                reverse=True,
            )[:3]

        # Sort by calibrated score (descending) before MMR
        paired.sort(key=lambda item: item[1], reverse=True)
        surviving_chunks = [c for c, _ in paired]
        surviving_scores = [s for _, s in paired]

        # Optionally annotate scores for transparency / debugging
        if return_scores:
            for chunk, score in paired:
                chunk.metadata["rerank_score"] = round(score, 4)
                chunk.metadata["rerank_task"] = task_type

        # MMR diversity pass
        embeddings: np.ndarray | None = None
        if self._bi_encoder is not None and self.lambda_mmr < 1.0 and len(surviving_chunks) > 1:
            try:
                raw_embs = self._bi_encoder.encode(
                    [c.text for c in surviving_chunks],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embeddings = raw_embs.astype("float32")
            except Exception:
                embeddings = None

        top_n = self.mmr_top_n or len(surviving_chunks)
        return _mmr_select(surviving_chunks, surviving_scores, embeddings, top_n=top_n, lambda_mmr=self.lambda_mmr)

