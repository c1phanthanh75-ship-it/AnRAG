from __future__ import annotations

import hashlib
from collections.abc import Iterable

from typing import Literal

from anrag.models import Chunk, ParsedBlock
from anrag.text import chunk_text_by_tokens, simple_tokens, split_sentences, token_count

ChunkingMode = Literal["fixed", "hierarchy", "semantic"]


def chunk_blocks(
    doc_id: str,
    blocks: list[ParsedBlock],
    mode: ChunkingMode = "semantic",
    max_tokens: int = 260,
    overlap_tokens: int = 40,
    min_semantic_tokens: int = 80,
    semantic_break_threshold: float = 0.12,
) -> list[Chunk]:
    if mode == "fixed":
        return fixed_chunking(doc_id, blocks, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if mode == "hierarchy":
        return fixed_hierarchical_chunking(
            doc_id,
            blocks,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
    return semantic_chunking(
        doc_id,
        blocks,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_semantic_tokens=min_semantic_tokens,
        semantic_break_threshold=semantic_break_threshold,
    )


def fixed_chunking(
    doc_id: str,
    blocks: list[ParsedBlock],
    max_tokens: int = 260,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Flat fixed-size token windows with no parent-child structure."""
    text = "\n\n".join(block.text for block in blocks if block.text.strip())
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    page_start = blocks[0].page if blocks else 1
    page_end = blocks[-1].page_end or blocks[-1].page if blocks else page_start

    for part_index, part in enumerate(chunk_text_by_tokens(text, max_tokens, overlap_tokens)):
        metadata = {"chunk_role": "fixed", "part_index": part_index}
        overlapping = [b for b in blocks if b.text.strip() and " ".join(b.text.split()) in part]
        hotpot_titles = []
        hotpot_sent_indices = []
        for b in overlapping:
            if "hotpot_title" in b.metadata and "hotpot_sent_idx" in b.metadata:
                hotpot_titles.append(b.metadata["hotpot_title"])
                hotpot_sent_indices.append(b.metadata["hotpot_sent_idx"])
        if hotpot_titles:
            metadata["hotpot_titles"] = hotpot_titles
            metadata["hotpot_sent_indices"] = hotpot_sent_indices

        chunks.append(
            Chunk(
                id=_chunk_id(doc_id, part_index, part),
                doc_id=doc_id,
                text=part,
                page_start=page_start,
                page_end=page_end,
                token_count=token_count(part),
                metadata=metadata,
            )
        )

    for index, chunk in enumerate(chunks):
        chunk.prev_id = chunks[index - 1].id if index > 0 else None
        chunk.next_id = chunks[index + 1].id if index + 1 < len(chunks) else None
    return chunks


def fixed_hierarchical_chunking(
    doc_id: str,
    blocks: list[ParsedBlock],
    max_tokens: int = 260,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Fixed token windows while preserving heading parent-child links."""
    chunks: list[Chunk] = []
    heading_chunk_by_block: dict[str, str] = {}

    def add_chunk(
        text: str,
        block: ParsedBlock,
        parent_id: str | None,
        role: str,
        part_index: int = 0,
        part_count: int = 1,
    ) -> Chunk:
        index = len(chunks)
        metadata = {
            "source_blocks": [block.id],
            "block_kind": block.kind,
            "chunk_role": role,
            "part_index": part_index,
            "part_count": part_count,
            **block.metadata,
        }
        if block.level is not None:
            metadata["heading_level"] = block.level
        chunk = Chunk(
            id=_chunk_id(doc_id, index, text),
            doc_id=doc_id,
            text=text,
            page_start=block.page,
            page_end=block.page_end or block.page,
            parent_id=parent_id,
            hierarchy_path=block.hierarchy_path,
            token_count=token_count(text),
            metadata=metadata,
        )
        chunks.append(chunk)
        return chunk

    def parent_chunk_id(block: ParsedBlock) -> str | None:
        if not block.parent_id:
            return None
        return heading_chunk_by_block.get(block.parent_id)

    for block in blocks:
        if block.kind == "heading":
            chunk = add_chunk(
                text=block.text,
                block=block,
                parent_id=parent_chunk_id(block),
                role="section",
            )
            heading_chunk_by_block[block.id] = chunk.id
            continue

        parent_id = parent_chunk_id(block)
        parts = chunk_text_by_tokens(block.text, max_tokens, overlap_tokens)
        for part_index, part in enumerate(parts):
            add_chunk(
                text=part,
                block=block,
                parent_id=parent_id,
                role="fixed",
                part_index=part_index,
                part_count=len(parts),
            )

    for index, chunk in enumerate(chunks):
        chunk.prev_id = chunks[index - 1].id if index > 0 else None
        chunk.next_id = chunks[index + 1].id if index + 1 < len(chunks) else None
    return chunks


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}|{index}|{text[:120]}".encode("utf-8")).hexdigest()[:16]
    return f"chk_{digest}"


def semantic_chunking(
    doc_id: str,
    blocks: list[ParsedBlock],
    max_tokens: int = 260,
    overlap_tokens: int = 40,
    min_semantic_tokens: int = 80,
    semantic_break_threshold: float = 0.12,
) -> list[Chunk]:
    """Chunk layout blocks by hierarchy, falling back to fixed token windows.

    Strategy order:
    1. Preserve heading chunks as section parents.
    2. Preserve tables/figures/scanned pages as layout-aware chunks.
    3. Split prose by paragraph and sentence with semantic breakpoints.
    4. Use fixed token windows only when an individual unit is too large.
    """

    chunks: list[Chunk] = []
    heading_chunk_by_block: dict[str, str] = {}

    def add_chunk(
        text: str,
        block: ParsedBlock,
        parent_id: str | None,
        role: str,
        part_index: int = 0,
        part_count: int = 1,
        extra_metadata: dict | None = None,
    ) -> Chunk:
        index = len(chunks)
        metadata = {
            "source_blocks": [block.id],
            "block_kind": block.kind,
            "chunk_role": role,
            "part_index": part_index,
            "part_count": part_count,
            **block.metadata,
        }
        if block.bbox:
            metadata["bbox"] = block.bbox
        if block.level is not None:
            metadata["heading_level"] = block.level
        if extra_metadata:
            metadata.update(extra_metadata)
        chunk = Chunk(
            id=_chunk_id(doc_id, index, text),
            doc_id=doc_id,
            text=text,
            page_start=block.page,
            page_end=block.page_end or block.page,
            parent_id=parent_id,
            hierarchy_path=block.hierarchy_path,
            token_count=token_count(text),
            metadata=metadata,
        )
        chunks.append(chunk)
        return chunk

    def parent_chunk_id(block: ParsedBlock) -> str | None:
        if not block.parent_id:
            return None
        return heading_chunk_by_block.get(block.parent_id)

    for block in blocks:
        if block.kind == "heading":
            chunk = add_chunk(
                text=block.text,
                block=block,
                parent_id=parent_chunk_id(block),
                role="section",
                extra_metadata={"heading_level": block.level},
            )
            heading_chunk_by_block[block.id] = chunk.id
            continue

        parent_id = parent_chunk_id(block)

        if block.kind == "table":
            parts = _split_table(block.text, max_tokens, overlap_tokens)
            for part_index, part in enumerate(parts):
                add_chunk(
                    text=part,
                    block=block,
                    parent_id=parent_id,
                    role="table",
                    part_index=part_index,
                    part_count=len(parts),
                    extra_metadata={"table_preserved": len(parts) == 1},
                )
            continue

        if block.kind in {"figure", "scanned_page"}:
            add_chunk(
                text=block.text,
                block=block,
                parent_id=parent_id,
                role=block.kind,
                extra_metadata={"requires_ocr": block.metadata.get("requires_ocr", False)},
            )
            continue

        prose_parts = _semantic_prose_chunks(
            block.text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_semantic_tokens=min_semantic_tokens,
            semantic_break_threshold=semantic_break_threshold,
        )
        for part_index, part in enumerate(prose_parts):
            add_chunk(
                text=part,
                block=block,
                parent_id=parent_id,
                role="prose",
                part_index=part_index,
                part_count=len(prose_parts),
            )

    for index, chunk in enumerate(chunks):
        chunk.prev_id = chunks[index - 1].id if index > 0 else None
        chunk.next_id = chunks[index + 1].id if index + 1 < len(chunks) else None

    return chunks


def _semantic_prose_chunks(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    min_semantic_tokens: int,
    semantic_break_threshold: float,
) -> list[str]:
    if token_count(text) <= max_tokens:
        return [text]

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return chunk_text_by_tokens(text, max_tokens, overlap_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_count(sentence)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.extend(chunk_text_by_tokens(sentence, max_tokens, overlap_tokens))
            continue

        should_break = False
        if current and current_tokens >= min_semantic_tokens:
            if current_tokens + sentence_tokens > max_tokens:
                should_break = True
            elif _lexical_similarity(current[-2:], [sentence]) < semantic_break_threshold:
                should_break = True

        if should_break:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk.strip()]


def _split_table(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if token_count(text) <= max_tokens:
        return [text]

    rows = [row for row in text.splitlines() if row.strip()]
    if len(rows) <= 1:
        return chunk_text_by_tokens(text, max_tokens, overlap_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    header = rows[0]
    header_tokens = token_count(header)
    for row in rows[1:]:
        row_tokens = token_count(row)
        if current and current_tokens + row_tokens > max_tokens:
            chunks.append("\n".join(current))
            current = [header]
            current_tokens = header_tokens
        if not current:
            current.append(header)
            current_tokens += header_tokens
        current.append(row)
        current_tokens += row_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


def _lexical_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_tokens = set(simple_tokens(" ".join(left)))
    right_tokens = set(simple_tokens(" ".join(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
