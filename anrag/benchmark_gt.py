from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from anrag.benchmark_types import BenchmarkFormat, BenchmarkQuestion
from anrag.models import ParsedBlock
from anrag.store import SQLiteTreeStore
from anrag.text_parsing import parse_plain_text

logger = logging.getLogger(__name__)


def document_id_for_corpus(corpus_key: str, namespace: str) -> str:
    digest = hashlib.sha1(f"{namespace}|{corpus_key}".encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def detect_benchmark_format(path: str | Path) -> BenchmarkFormat:
    path = Path(path)
    if path.is_dir():
        if (path / "corpus.jsonl").exists() and (path / "queries.jsonl").exists():
            return "beir"
        qrels = path / "qrels"
        if qrels.is_dir() and any(qrels.glob("*.tsv")):
            return "beir"

    if path.is_file():
        if path.suffix.lower() == ".tsv" and "qrels" in path.as_posix():
            return "beir"
        sample = _read_json_sample(path)
        if sample:
            if "supporting_facts" in sample and "context" in sample:
                return "hotpotqa"
            if "provenance" in sample and "input" in sample:
                return "kilt"
            if "gold_chunk_ids" in sample or "question" in sample:
                return "anrag"

    return "anrag"


def _read_json_sample(path: Path) -> dict | None:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line)
        return None
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
    return None


def parse_hotpot_context(
    context: list[list[object]],
    *,
    doc_id: str,
) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    for entry in context:
        title = str(entry[0])
        sentences = [str(item) for item in entry[1]]
        heading_id = f"hdr_{hashlib.sha1(f'{doc_id}|{title}'.encode()).hexdigest()[:16]}"
        blocks.append(
            ParsedBlock(
                id=heading_id,
                text=title,
                page=1,
                kind="heading",
                level=2,
                hierarchy_path=[title],
                metadata={"hotpot_title": title, "layout_role": "heading"},
            )
        )
        for sent_idx, sentence in enumerate(sentences):
            blocks.append(
                ParsedBlock(
                    id=f"para_{hashlib.sha1(f'{doc_id}|{title}|{sent_idx}|{sentence[:80]}'.encode()).hexdigest()[:16]}",
                    text=sentence,
                    page=1,
                    kind="paragraph",
                    parent_id=heading_id,
                    hierarchy_path=[title],
                    metadata={
                        "hotpot_title": title,
                        "hotpot_sent_idx": sent_idx,
                        "hotpot_sentence": sentence,
                        "layout_role": "paragraph",
                    },
                )
            )
    return blocks


def load_hotpotqa(path: str | Path) -> tuple[dict[str, list[ParsedBlock]], list[BenchmarkQuestion]]:
    path = Path(path)
    items = _load_json_items(path)
    documents: dict[str, list[ParsedBlock]] = {}
    questions: list[BenchmarkQuestion] = []

    for item in items:
        qid = str(item.get("_id") or item.get("id") or len(questions))
        doc_id = document_id_for_corpus(qid, "hotpotqa")
        documents[doc_id] = parse_hotpot_context(item["context"], doc_id=doc_id)
        questions.append(
            BenchmarkQuestion(
                question=item["question"],
                doc_id=doc_id,
                query_id=qid,
                gold_passage_keys=[(str(title), int(idx)) for title, idx in item["supporting_facts"]],
                benchmark_format="hotpotqa",
            )
        )
    return documents, questions


def load_beir(dataset_dir: str | Path) -> tuple[dict[str, list[ParsedBlock]], list[BenchmarkQuestion]]:
    root = Path(dataset_dir)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = _find_beir_qrels(root)
    if not corpus_path.exists() or not queries_path.exists() or qrels_path is None:
        raise FileNotFoundError(
            f"BeIR layout expected corpus.jsonl, queries.jsonl, qrels/*.tsv under {root}"
        )

    documents: dict[str, list[ParsedBlock]] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        corpus_id = str(row["_id"])
        doc_id = document_id_for_corpus(corpus_id, "beir")
        title = str(row.get("title") or "")
        body = str(row.get("text") or "")
        text = f"{title}\n\n{body}".strip() if title else body
        documents[doc_id] = parse_plain_text(
            text,
            doc_id=doc_id,
            extra_metadata={"beir_corpus_id": corpus_id, "beir_title": title},
        )

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            qrels.setdefault(query_id, {})[corpus_id] = score

    questions: list[BenchmarkQuestion] = []
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row["_id"])
        rels = qrels.get(query_id, {})
        gold_doc_ids = {document_id_for_corpus(corpus_id, "beir") for corpus_id in rels}
        relevance_grades = {
            document_id_for_corpus(corpus_id, "beir"): float(score)
            for corpus_id, score in rels.items()
        }
        questions.append(
            BenchmarkQuestion(
                question=row.get("text") or row.get("query") or "",
                query_id=query_id,
                gold_doc_ids=gold_doc_ids,
                relevance_grades=relevance_grades,
                benchmark_format="beir",
            )
        )
    return documents, questions


def load_kilt(path: str | Path) -> tuple[dict[str, list[ParsedBlock]], list[BenchmarkQuestion]]:
    path = Path(path)
    items = _load_json_items(path)
    documents: dict[str, list[ParsedBlock]] = {}
    questions: list[BenchmarkQuestion] = []

    for item in items:
        qid = str(item.get("id") or len(questions))
        provenance = item.get("provenance") or []
        doc_id = document_id_for_corpus(qid, "kilt")
        blocks: list[ParsedBlock] = []
        for prov_index, prov in enumerate(provenance):
            title = str(prov.get("title") or f"Section {prov_index + 1}")
            section = str(prov.get("section") or "")
            wiki_id = str(prov.get("wikipedia_id") or "")
            evidence = str(prov.get("meta", {}).get("evidence") or prov.get("text") or section or title)
            heading = f"{title} — {section}".strip(" —")
            blocks.extend(
                parse_plain_text(
                    evidence,
                    doc_id=doc_id,
                    extra_metadata={
                        "kilt_wikipedia_id": wiki_id,
                        "kilt_title": title,
                        "kilt_section": section,
                        "kilt_prov_index": prov_index,
                    },
                )
            )
        if not blocks:
            blocks = parse_plain_text(str(item.get("input") or ""), doc_id=doc_id)
        documents[doc_id] = blocks

        gold_doc_ids = {doc_id}
        gold_passages = [
            str(prov.get("meta", {}).get("evidence") or prov.get("text") or prov.get("section") or "")
            for prov in provenance
            if prov
        ]
        questions.append(
            BenchmarkQuestion(
                question=str(item.get("input") or ""),
                doc_id=doc_id,
                query_id=qid,
                gold_doc_ids=gold_doc_ids,
                gold_passages=[passage for passage in gold_passages if passage.strip()],
                benchmark_format="kilt",
            )
        )
    return documents, questions


def load_anrag_questions(path: str | Path) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            BenchmarkQuestion(
                question=row["question"],
                doc_id=row.get("doc_id"),
                query_id=row.get("query_id"),
                gold_chunk_ids=set(row.get("gold_chunk_ids", [])),
                benchmark_format="anrag",
            )
        )
    return questions


def load_official_benchmark(
    path: str | Path,
    fmt: BenchmarkFormat | None = None,
) -> tuple[dict[str, list[ParsedBlock]], list[BenchmarkQuestion]]:
    path = Path(path)
    fmt = fmt or detect_benchmark_format(path)
    if fmt == "hotpotqa":
        return load_hotpotqa(path)
    if fmt == "beir":
        return load_beir(path)
    if fmt == "kilt":
        return load_kilt(path)
    raise ValueError(f"Path {path} is not an official benchmark dataset (format={fmt})")


def resolve_gold_official(question, store: SQLiteTreeStore) -> set[str]:
    if question.gold_chunk_ids:
        return set(question.gold_chunk_ids)

    if question.gold_doc_ids:
        gold: set[str] = set()
        for doc_id in question.gold_doc_ids:
            gold.update(chunk.id for chunk in store.all_chunks(doc_id))
        return gold

    if question.gold_passage_keys:
        return _resolve_hotpot_passages(question, store)

    if question.gold_passages:
        return _resolve_passage_texts(question, store)

    if question.benchmark_format != "anrag":
        logger.warning(
            "No official gold resolved for query %s (%s)",
            question.query_id or question.question[:40],
            question.benchmark_format,
        )
    return set()


def build_relevance_map(question, store: SQLiteTreeStore, gold_ids: set[str]) -> dict[str, float]:
    if question.relevance_grades:
        mapped: dict[str, float] = {}
        for chunk in store.all_chunks():
            corpus_id = chunk.metadata.get("beir_corpus_id")
            if corpus_id:
                doc_key = document_id_for_corpus(str(corpus_id), "beir")
                mapped[chunk.id] = float(question.relevance_grades.get(doc_key, 0.0))
            elif chunk.doc_id in question.relevance_grades:
                mapped[chunk.id] = float(question.relevance_grades[chunk.doc_id])
        if mapped:
            return mapped
    return {chunk_id: 1.0 for chunk_id in gold_ids}


def _resolve_hotpot_passages(question, store: SQLiteTreeStore) -> set[str]:
    gold: set[str] = set()
    chunks = store.all_chunks(question.doc_id) if question.doc_id else store.all_chunks()
    key_set = {(title, idx) for title, idx in question.gold_passage_keys}
    for chunk in chunks:
        title = chunk.metadata.get("hotpot_title")
        sent_idx = chunk.metadata.get("hotpot_sent_idx")
        if title is not None and sent_idx is not None and (str(title), int(sent_idx)) in key_set:
            gold.add(chunk.id)
    return gold


def _resolve_passage_texts(question, store: SQLiteTreeStore) -> set[str]:
    gold: set[str] = set()
    chunks = store.all_chunks(question.doc_id) if question.doc_id else store.all_chunks()
    targets = [text.strip().lower() for text in question.gold_passages if text.strip()]
    for chunk in chunks:
        body = chunk.text.strip().lower()
        if any(target in body or body in target for target in targets):
            gold.add(chunk.id)
    return gold


def _find_beir_qrels(root: Path) -> Path | None:
    qrels_dir = root / "qrels"
    if qrels_dir.is_dir():
        candidates = sorted(qrels_dir.glob("*.tsv"))
        if candidates:
            return candidates[0]
    direct = sorted(root.glob("qrels*.tsv"))
    return direct[0] if direct else None


def _load_json_items(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [payload]
