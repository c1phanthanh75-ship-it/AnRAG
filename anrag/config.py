from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("data"))
    sqlite_path: Path = Field(default=Path("data/anrag.sqlite3"))
    index_dir: Path = Field(default=Path("data/indexes"))
    upload_dir: Path = Field(default=Path("data/uploads"))
    visual_dir: Path = Field(default=Path("data/visuals"))

    ollama_model: str = "qwen3:8b"
    ollama_host: str | None = None
    vision_model: str | None = None

    embedding_mode: str = "auto"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    max_chunk_tokens: int = 260
    chunk_overlap_tokens: int = 40
    min_semantic_chunk_tokens: int = 80
    semantic_break_threshold: float = 0.12
    context_budget_tokens: int = 1200
    anchor_confidence_threshold: float = 0.32

    # Hybrid retrieval fusion weights (dense + sparse via RRF).
    # dense_weight + sparse_weight need not sum to 1; they are relative.
    # Increase sparse_weight for keyword-heavy corpora (e.g. legal, code).
    # Increase dense_weight for natural-language / semantic queries.
    hybrid_dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    hybrid_sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    # RRF constant k (Cormack et al., 2009). 60 is the standard default.
    hybrid_rrf_k: int = Field(default=60, ge=1)

    # Cross-encoder reranker diversity (MMR).
    # lambda_mmr=1.0 → pure relevance; 0.5 → balanced diversity/relevance.
    reranker_lambda_mmr: float = Field(default=0.5, ge=0.0, le=1.0)
    # Maximum chunks kept after MMR pass (None = keep all survivors).
    reranker_mmr_top_n: int | None = None

    enable_ocr: bool = True
    ocr_lang: str | None = "en"
    ocr_min_confidence: float = 0.35
    ocr_render_dpi: int = 180
    ocr_text_detection_model_dir: str | None = None
    ocr_text_recognition_model_dir: str | None = None
    enable_vision_caption: bool = False

    class Config:
        env_prefix = "ANRAG_"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def benchmark_eval_settings(base: Settings | None = None) -> Settings:
    """Settings for retrieval-only benchmark runs without LLM, VLM, or OCR noise."""
    base_settings = base or get_settings()
    settings = base_settings.model_copy(
        update={
            "enable_ocr": False,
            "enable_vision_caption": False,
            "data_dir": Path("data/benchmark"),
            "sqlite_path": Path("data/benchmark/anrag.sqlite3"),
            "index_dir": Path("data/benchmark/indexes"),
            "upload_dir": Path("data/benchmark/uploads"),
            "visual_dir": Path("data/benchmark/visuals"),
        }
    )
    settings.ensure_dirs()
    return settings
