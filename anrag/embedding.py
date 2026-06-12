from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger(__name__)

# Quality tiers are exposed on the instance so callers (e.g. retrieval,
# trace output) can surface them to the user without knowing internals.
QUALITY_TIER_SEMANTIC = "sentence-transformers"
QUALITY_TIER_HASHING = "hashing"

_HASHING_WARNING = (
    "EmbeddingBackend falling back to HashingVectorizer (TF-IDF-like hashing). "
    "Retrieval quality will be significantly lower than with a real sentence-transformer model. "
    "Install sentence-transformers and ensure the model '{model}' is accessible to restore "
    "semantic search quality."
)


class EmbeddingBackend:
    """Dual-mode embedding backend: sentence-transformers (preferred) or HashingVectorizer.

    Attributes
    ----------
    quality_tier : str
        Either ``QUALITY_TIER_SEMANTIC`` or ``QUALITY_TIER_HASHING``.
        Downstream components can check this to decide whether to warn
        users or adjust confidence thresholds.
    """

    def __init__(self, mode: str = "auto", model_name: str | None = None, dim: int = 384):
        self.mode = mode
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._hashing = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            token_pattern=r"(?u)\b\w+\b",
        )

        if mode in {"auto", "sentence-transformers"}:
            try:
                from sentence_transformers import SentenceTransformer

                resolved_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self._model = SentenceTransformer(resolved_name)
                self.mode = "sentence-transformers"
                self.quality_tier = QUALITY_TIER_SEMANTIC
            except Exception as exc:
                if mode == "sentence-transformers":
                    # Hard mode: propagate — the user explicitly requested this backend.
                    raise
                # Auto mode: degrade gracefully but always warn loudly.
                resolved_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                msg = _HASHING_WARNING.format(model=resolved_name)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                logger.warning("EmbeddingBackend degraded to hashing: %s", exc)
                self.mode = "hashing"
                self.quality_tier = QUALITY_TIER_HASHING
        else:
            self.mode = "hashing"
            self.quality_tier = QUALITY_TIER_HASHING
            logger.info(
                "EmbeddingBackend initialized in hashing mode (explicit). "
                "Consider switching to sentence-transformers for better retrieval quality."
            )

    @property
    def is_semantic(self) -> bool:
        """True if backed by a real sentence-transformer model."""
        return self.quality_tier == QUALITY_TIER_SEMANTIC

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if self.mode == "sentence-transformers" and self._model is not None:
            vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype("float32")
        vectors = self._hashing.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
