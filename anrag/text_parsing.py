from __future__ import annotations

import hashlib
import re

from anrag.models import ParsedBlock
from anrag.text import normalize_text

_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(\S.+)$")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _heading_level(line: str) -> int | None:
    stripped = line.strip()
    if not stripped:
        return None

    markdown = _MARKDOWN_HEADING_RE.match(stripped)
    if markdown:
        return len(markdown.group(1))

    numbered = _NUMBERED_HEADING_RE.match(stripped)
    if numbered:
        return min(6, numbered.group(1).count(".") + 1)

    if stripped.isupper() and len(stripped.split()) <= 12 and len(stripped) < 100:
        return 2

    return None


def _heading_text(line: str) -> str:
    markdown = _MARKDOWN_HEADING_RE.match(line.strip())
    if markdown:
        return normalize_text(markdown.group(2))
    numbered = _NUMBERED_HEADING_RE.match(line.strip())
    if numbered:
        return normalize_text(f"{numbered.group(1)} {numbered.group(2)}")
    return normalize_text(line.strip())


def _attach_hierarchy(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    heading_stack: list[ParsedBlock] = []
    result: list[ParsedBlock] = []
    for block in blocks:
        if block.kind == "heading":
            while heading_stack and (heading_stack[-1].level or 0) >= (block.level or 1):
                heading_stack.pop()
            parent = heading_stack[-1] if heading_stack else None
            block.parent_id = parent.id if parent else None
            block.hierarchy_path = [item.text for item in heading_stack] + [block.text]
            heading_stack.append(block)
        else:
            parent = heading_stack[-1] if heading_stack else None
            block.parent_id = parent.id if parent else None
            block.hierarchy_path = [item.text for item in heading_stack]
        result.append(block)
    return result


def parse_plain_text(
    text: str,
    *,
    doc_id: str = "doc_text",
    page: int = 1,
    extra_metadata: dict | None = None,
) -> list[ParsedBlock]:
    """Convert plain text into heading/paragraph ``ParsedBlock`` objects."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[ParsedBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        body = normalize_text(" ".join(paragraph_lines))
        if not body:
            paragraph_lines.clear()
            return
        metadata = {"layout_role": "paragraph", "source_format": "plain_text"}
        if extra_metadata:
            metadata.update(extra_metadata)
        blocks.append(
            ParsedBlock(
                id=_stable_id("para", doc_id, len(blocks), body[:100]),
                text=body,
                page=page,
                kind="paragraph",
                metadata=metadata,
            )
        )
        paragraph_lines.clear()

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        level = _heading_level(raw_line)
        if level is not None:
            flush_paragraph()
            heading = _heading_text(raw_line)
            blocks.append(
                ParsedBlock(
                    id=_stable_id("hdr", doc_id, len(blocks), heading),
                    text=heading,
                    page=page,
                    kind="heading",
                    level=level,
                    metadata={"layout_role": "heading", "source_format": "plain_text"},
                )
            )
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return _attach_hierarchy(blocks)
