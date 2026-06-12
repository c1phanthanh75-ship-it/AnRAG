from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BenchmarkFormat = Literal["anrag", "hotpotqa", "beir", "kilt", "ragbench"]


@dataclass
class BenchmarkQuestion:
    question: str
    doc_id: str | None = None
    query_id: str | None = None
    gold_chunk_ids: set[str] = field(default_factory=set)
    gold_doc_ids: set[str] = field(default_factory=set)
    gold_passages: list[str] = field(default_factory=list)
    gold_passage_keys: list[tuple[str, int]] = field(default_factory=list)
    relevance_grades: dict[str, float] = field(default_factory=dict)
    benchmark_format: BenchmarkFormat = "anrag"
