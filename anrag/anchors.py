from __future__ import annotations

import re

from anrag.models import Chunk


_DEFINITION_PATTERNS = [
    r"\bis defined as\b",
    r"\brefers to\b",
    r"\bmeans\b",
    r"\blà\b",
    r"\bđược định nghĩa\b",
    r"\bgọi là\b",
    r"\bcó nghĩa là\b",
]
_THEOREM_PATTERNS = [
    r"\btheorem\b",
    r"\blemma\b",
    r"\bproposition\b",
    r"\bcorollary\b",
    r"\bđịnh lý\b",
    r"\bbổ đề\b",
    r"\bhệ quả\b",
]
_METHOD_PATTERNS = [
    r"\balgorithm\b",
    r"\bmethod\b",
    r"\bapproach\b",
    r"\bpipeline\b",
    r"\bframework\b",
    r"\bphương pháp\b",
    r"\bthuật toán\b",
    r"\bquy trình\b",
    r"\bmô hình\b",
]
_OVERVIEW_PATTERNS = [
    r"\bwe propose\b",
    r"\bwe introduce\b",
    r"\bthis section\b",
    r"\bin this paper\b",
    r"\bchúng tôi đề xuất\b",
    r"\bphần này\b",
    r"\bbài báo này\b",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def mark_anchors(chunks: list[Chunk]) -> list[Chunk]:
    for index, chunk in enumerate(chunks):
        text = chunk.text.strip()
        lower = text.lower()
        heading_level = chunk.metadata.get("heading_level")

        if heading_level == 1 or (index == 0 and len(text) <= 160):
            chunk.anchor_type = "TITLE"
        elif heading_level is not None and _matches(_METHOD_PATTERNS, lower):
            chunk.anchor_type = "METHOD"
        elif heading_level is not None:
            chunk.anchor_type = "OVERVIEW"
        elif _matches(_THEOREM_PATTERNS, lower):
            chunk.anchor_type = "THEOREM"
        elif _matches(_DEFINITION_PATTERNS, lower):
            chunk.anchor_type = "DEFINITION"
        elif len(text.split()) <= 90 and _matches(_OVERVIEW_PATTERNS, lower):
            chunk.anchor_type = "OVERVIEW"
        elif _matches(_METHOD_PATTERNS, lower) and len(text.split()) <= 140:
            chunk.anchor_type = "METHOD"

    return chunks
