from __future__ import annotations

import hashlib
from pathlib import Path


def document_id_for_file(path: str | Path) -> str:
    path = Path(path)
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
    return "doc_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
