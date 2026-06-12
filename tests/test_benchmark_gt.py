import json
from pathlib import Path

from anrag.benchmark import BenchmarkParser
from anrag.benchmark_gt import (
    detect_benchmark_format,
    load_beir,
    load_hotpotqa,
    load_kilt,
    resolve_gold_official,
)
from anrag.benchmark_types import BenchmarkQuestion
from anrag.chunking import semantic_chunking
from anrag.pipeline import ingest_blocks
from anrag.store import SQLiteTreeStore
from anrag.config import Settings


def test_parse_text_builds_heading_hierarchy():
    text = "# Introduction\n\nAnchor retrieval improves context.\n\n## Method\n\nTree expansion helps."
    blocks = BenchmarkParser().parse_text(text, doc_id="doc_txt")
    assert any(block.kind == "heading" for block in blocks)
    assert any(block.kind == "paragraph" and block.parent_id for block in blocks)


def test_detect_hotpotqa_format(tmp_path):
    path = tmp_path / "hotpot.jsonl"
    path.write_text(
        json.dumps(
            {
                "_id": "1",
                "question": "Who founded it?",
                "context": [["Acme", ["Acme was founded in 1999.", "It sells tools."]]],
                "supporting_facts": [["Acme", 0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert detect_benchmark_format(path) == "hotpotqa"
    documents, questions = load_hotpotqa(path)
    assert "doc_" in next(iter(documents))
    assert questions[0].gold_passage_keys == [("Acme", 0)]


def test_hotpotqa_official_gold_resolution(tmp_path):
    path = tmp_path / "hotpot.jsonl"
    path.write_text(
        json.dumps(
            {
                "_id": "1",
                "question": "When was Acme founded?",
                "context": [["Acme", ["Acme was founded in 1999."]]],
                "supporting_facts": [["Acme", 0]],
            }
        ),
        encoding="utf-8",
    )
    documents, questions = load_hotpotqa(path)
    settings = Settings(
        sqlite_path=tmp_path / "gt.sqlite3",
        index_dir=tmp_path / "indexes",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        visual_dir=tmp_path / "visuals",
    )
    settings.ensure_dirs()
    store = SQLiteTreeStore(settings.sqlite_path)
    doc_id = next(iter(documents))
    ingest_blocks(doc_id, "hotpot.jsonl", "hotpot.jsonl", documents[doc_id], settings, store)
    gold = resolve_gold_official(questions[0], store)
    assert len(gold) == 1


def test_load_beir_layout(tmp_path):
    root = tmp_path / "fiqa"
    root.mkdir()
    (root / "corpus.jsonl").write_text(
        '{"_id": "d1", "title": "Banking", "text": "Interest rates moved higher."}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text('{"_id": "q1", "text": "interest rates"}\n', encoding="utf-8")
    qrels = root / "qrels"
    qrels.mkdir()
    (qrels / "test.tsv").write_text("q1\td1\t2\n", encoding="utf-8")

    documents, questions = load_beir(root)
    assert len(documents) == 1
    assert questions[0].benchmark_format == "beir"
    assert questions[0].gold_doc_ids


def test_load_kilt_provenance(tmp_path):
    path = tmp_path / "kilt.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "k1",
                "input": "Who is Ada Lovelace?",
                "provenance": [
                    {
                        "wikipedia_id": "123",
                        "title": "Ada Lovelace",
                        "section": "Biography",
                        "meta": {"evidence": "Ada Lovelace was an English mathematician."},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    documents, questions = load_kilt(path)
    assert questions[0].benchmark_format == "kilt"
    assert questions[0].gold_passages
