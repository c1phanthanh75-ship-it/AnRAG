from __future__ import annotations

from pathlib import Path

from typing import Literal

from anrag.anchors import mark_anchors
from anrag.benchmark import BenchmarkParser
from anrag.chunking import ChunkingMode, chunk_blocks
from anrag.config import Settings
from anrag.documents import document_id_for_file
from anrag.llm import OllamaLLM
from anrag.models import ParsedBlock
from anrag.parsing import parse_pdf
from anrag.store import SQLiteTreeStore

RetrievalMode = Literal["baseline", "anchor_only", "full"]


def ingest_blocks(
    doc_id: str,
    name: str,
    source_path: str,
    blocks: list[ParsedBlock],
    settings: Settings,
    store: SQLiteTreeStore,
    *,
    chunking_mode: ChunkingMode = "semantic",
    mark_anchor_tags: bool = True,
) -> str:
    chunks = chunk_blocks(
        doc_id=doc_id,
        blocks=blocks,
        mode=chunking_mode,
        max_tokens=settings.max_chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_semantic_tokens=settings.min_semantic_chunk_tokens,
        semantic_break_threshold=settings.semantic_break_threshold,
    )
    if mark_anchor_tags:
        mark_anchors(chunks)
    store.upsert_document(doc_id, name, source_path)
    store.replace_chunks(doc_id, chunks)
    return doc_id


def ingest_pdf(path: str | Path, settings: Settings, store: SQLiteTreeStore, llm: OllamaLLM | None = None) -> str:
    path = Path(path)
    doc_id = document_id_for_file(path)
    blocks = parse_pdf(
        path,
        enable_ocr=settings.enable_ocr,
        ocr_lang=settings.ocr_lang,
        ocr_min_confidence=settings.ocr_min_confidence,
        ocr_render_dpi=settings.ocr_render_dpi,
        ocr_text_detection_model_dir=settings.ocr_text_detection_model_dir,
        ocr_text_recognition_model_dir=settings.ocr_text_recognition_model_dir,
        visual_dir=settings.visual_dir,
        enable_vision_caption=settings.enable_vision_caption,
        llm=llm,
    )
    return ingest_blocks(
        doc_id,
        path.name,
        str(path),
        blocks,
        settings,
        store,
        chunking_mode="semantic",
        mark_anchor_tags=True,
    )


def ingest_benchmark(
    path: str | Path,
    settings: Settings,
    store: SQLiteTreeStore,
    llm: OllamaLLM | None = None,
    *,
    chunking_mode: ChunkingMode = "semantic",
    mark_anchor_tags: bool = True,
) -> list[str]:
    parser = BenchmarkParser()
    documents = parser.parse_benchmark(path, settings=settings, llm=llm)
    doc_ids: list[str] = []
    benchmark_path = Path(path)
    for doc_id, blocks in documents.items():
        if benchmark_path.is_dir():
            matches = [pdf for pdf in benchmark_path.rglob("*.pdf") if document_id_for_file(pdf) == doc_id]
            source = str(matches[0]) if matches else str(benchmark_path / doc_id)
        else:
            source = str(benchmark_path)
        ingest_blocks(
            doc_id,
            Path(source).name,
            source,
            blocks,
            settings,
            store,
            chunking_mode=chunking_mode,
            mark_anchor_tags=mark_anchor_tags,
        )
        doc_ids.append(doc_id)
    return doc_ids
