from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|(?<=\.)\s+(?=[A-ZÀ-Ỹ])")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_query(query: str) -> str:
    return normalize_text(query).strip(" ?!.;:")


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def token_count(text: str) -> int:
    return len(simple_tokens(text))


def chunk_text_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(tokens), step):
        part = tokens[start : start + max_tokens]
        if part:
            chunks.append(" ".join(part))
        if start + max_tokens >= len(tokens):
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    pieces = [normalize_text(piece) for piece in _SENTENCE_RE.split(text)]
    return [piece for piece in pieces if piece]
