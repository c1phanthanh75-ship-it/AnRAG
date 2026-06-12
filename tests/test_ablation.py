from pathlib import Path

import pytest

from anrag.ablation import compute_contributions, run_ablations_from_documents
from anrag.anchors import mark_anchors
from anrag.benchmark import BenchmarkParser
from anrag.chunking import fixed_chunking, fixed_hierarchical_chunking, semantic_chunking
from anrag.config import Settings
from anrag.models import ParsedBlock


def _sample_blocks() -> list[ParsedBlock]:
    return [
        ParsedBlock(id="h1", text="1 Introduction", page=1, kind="heading", level=1, hierarchy_path=["1 Introduction"]),
        ParsedBlock(
            id="p1",
            text="Anchor retrieval finds section-level signals before expanding local context.",
            page=1,
            kind="paragraph",
            parent_id="h1",
            hierarchy_path=["1 Introduction"],
        ),
        ParsedBlock(
            id="h2",
            text="2 Method",
            page=2,
            kind="heading",
            level=2,
            parent_id="h1",
            hierarchy_path=["1 Introduction", "2 Method"],
        ),
        ParsedBlock(
            id="p2",
            text="The method is defined as anchor-first retrieval with tree expansion.",
            page=2,
            kind="paragraph",
            parent_id="h2",
            hierarchy_path=["1 Introduction", "2 Method"],
        ),
    ]


def test_benchmark_parser_jsonl_still_reserved():
    parser = BenchmarkParser()
    with pytest.raises(NotImplementedError):
        parser.parse_jsonl("future.jsonl")


def test_fixed_chunking_has_no_parent_links():
    chunks = fixed_chunking("doc_test", _sample_blocks(), max_tokens=12, overlap_tokens=2)
    assert chunks
    assert all(chunk.parent_id is None for chunk in chunks)
    assert all(chunk.metadata["chunk_role"] == "fixed" for chunk in chunks)


def test_fixed_hierarchical_chunking_keeps_parent_child_links():
    chunks = fixed_hierarchical_chunking("doc_test", _sample_blocks(), max_tokens=12, overlap_tokens=2)
    section = next(chunk for chunk in chunks if chunk.text == "1 Introduction")
    children = [chunk for chunk in chunks if chunk.parent_id == section.id]
    fixed_children = [chunk for chunk in children if chunk.metadata["chunk_role"] == "fixed"]
    assert fixed_children
    assert any(chunk.metadata["chunk_role"] == "section" for chunk in children)


def test_compute_contributions_splits_total_gain():
    result = compute_contributions(
        baseline_score=0.20,
        hierarchy_score=0.35,
        anchor_only_score=0.55,
        full_score=0.80,
    )
    assert result.hierarchy_pct == 25.0
    assert result.anchor_pct == 33.33
    assert result.expansion_pct == 41.67
    assert round(result.hierarchy_pct + result.anchor_pct + result.expansion_pct, 2) == 100.0


def test_run_ablations_from_documents_reports_metrics(tmp_path):
    blocks = _sample_blocks()
    settings = Settings(
        sqlite_path=tmp_path / "ablation.sqlite3",
        index_dir=tmp_path / "indexes",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        visual_dir=tmp_path / "visuals",
    )
    settings.ensure_dirs()

    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        '{"question": "Anchor retrieval finds section-level", "doc_id": "doc_ablation", '
        '"gold_passages": ["Anchor retrieval finds section-level signals"]}\n',
        encoding="utf-8",
    )

    report = run_ablations_from_documents(
        {"doc_ablation": blocks},
        qa_path=qa_path,
        settings=settings,
        sqlite_path=settings.sqlite_path,
        top_k=4,
        budget_tokens=400,
    )

    assert len(report.scores) == 4
    full_score = next(score for score in report.scores if score.key == "full")
    assert full_score.metrics.recall_at_k >= 0.0
    assert full_score.metrics.recall_at_5 >= 0.0
    assert full_score.metrics.recall_at_10 >= 0.0
    assert full_score.metrics.recall_at_20 >= 0.0
    assert full_score.metrics.mrr >= 0.0
    assert full_score.metrics.ndcg_at_10 >= 0.0
    assert report.time_cost.total_seconds > 0

    payload = report.to_dict()
    assert payload["summary"]["primary_metric"] == "recall_at_k"
    assert "recall_at_10" in payload["summary"]["full_anrag_metrics"]


def test_full_anrag_chunking_marks_anchors():
    chunks = mark_anchors(semantic_chunking("doc_test", _sample_blocks(), max_tokens=80))
    assert any(chunk.anchor_type for chunk in chunks)
