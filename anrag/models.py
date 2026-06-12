from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Iterable


AnchorType = Literal["TITLE", "OVERVIEW", "DEFINITION", "THEOREM", "METHOD"]
BlockKind = Literal["heading", "paragraph", "table", "figure", "scanned_page"]


@dataclass
class ParsedBlock:
    id: str
    text: str
    page: int
    kind: BlockKind
    level: int | None = None
    parent_id: str | None = None
    hierarchy_path: list[str] = field(default_factory=list)
    font_size: float = 0.0
    page_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    parent_id: str | None = None
    prev_id: str | None = None
    next_id: str | None = None
    hierarchy_path: list[str] = field(default_factory=list)
    token_count: int = 0
    anchor_type: AnchorType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anchor(self) -> bool:
        return self.anchor_type is not None


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    source: str


@dataclass
class RetrievalResult:
    answer: str | Iterable[str]
    contexts: list[Chunk]
    anchors: list[SearchHit]
    trace: dict[str, Any]
