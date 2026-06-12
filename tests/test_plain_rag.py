from pathlib import Path

from anrag.anchors import mark_anchors
from anrag.chunking import semantic_chunking
from anrag.config import Settings, benchmark_eval_settings
from anrag.embedding import EmbeddingBackend
from anrag.models import ParsedBlock
from anrag.retrieval import AnchorRetriever
from anrag.store import SQLiteTreeStore


def _blocks() -> list[ParsedBlock]:
    return [
        ParsedBlock(id="h1", text="AnchorRAG Overview", page=1, kind="heading", level=1, hierarchy_path=["AnchorRAG Overview"]),
        ParsedBlock(
            id="p1",
            text="Hybrid retrieval combines dense embeddings and BM25 sparse search.",
            page=1,
            kind="paragraph",
            parent_id="h1",
            hierarchy_path=["AnchorRAG Overview"],
        ),
        ParsedBlock(
            id="p2",
            text="Anchor expansion adds parent chunks, siblings, and local neighbors.",
            page=1,
            kind="paragraph",
            parent_id="h1",
            hierarchy_path=["AnchorRAG Overview"],
        ),
    ]


def test_plain_rag_uses_hybrid_chunk_retrieval(tmp_path):
    settings = Settings(
        sqlite_path=tmp_path / "plain.sqlite3",
        index_dir=tmp_path / "indexes",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        visual_dir=tmp_path / "visuals",
    )
    settings.ensure_dirs()
    store = SQLiteTreeStore(settings.sqlite_path)
    chunks = mark_anchors(semantic_chunking("doc_plain", _blocks(), max_tokens=80))
    store.upsert_document("doc_plain", "plain.pdf", "plain.pdf")
    store.replace_chunks("doc_plain", chunks)

    retriever = AnchorRetriever(
        store,
        str(settings.index_dir),
        EmbeddingBackend("hashing", settings.embedding_model, settings.embedding_dim),
        llm=None,
        rrf_k=settings.hybrid_rrf_k,
    )
    retriever.rebuild_indexes()

    result = retriever.plain_rag(
        "hybrid retrieval BM25",
        top_k=4,
        budget_tokens=400,
        generate_answer=False,
        rewrite_query=False,
    )

    assert result.trace["mode"] == "plain_rag"
    assert result.trace["fusion_mode"] in {"rrf", "dense_only"}
    assert result.contexts


def test_benchmark_eval_settings_disable_ocr_and_vlm():
    settings = benchmark_eval_settings(Settings(enable_ocr=True, enable_vision_caption=True))
    assert settings.enable_ocr is False
    assert settings.enable_vision_caption is False
