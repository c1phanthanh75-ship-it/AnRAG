from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from anrag.models import Chunk


class SQLiteTreeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                create table if not exists documents (
                    id text primary key,
                    name text not null,
                    path text not null,
                    created_at real not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists chunks (
                    id text primary key,
                    doc_id text not null,
                    text text not null,
                    page_start integer not null,
                    page_end integer not null,
                    parent_id text,
                    prev_id text,
                    next_id text,
                    hierarchy_path text not null,
                    token_count integer not null,
                    anchor_type text,
                    metadata text not null,
                    foreign key(doc_id) references documents(id)
                )
                """
            )
            conn.execute("create index if not exists idx_chunks_doc on chunks(doc_id)")
            conn.execute("create index if not exists idx_chunks_parent on chunks(parent_id)")
            conn.execute("create index if not exists idx_chunks_anchor on chunks(anchor_type)")

    def upsert_document(self, doc_id: str, name: str, path: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into documents(id, name, path, created_at)
                values(?, ?, ?, ?)
                on conflict(id) do update set
                  name=excluded.name,
                  path=excluded.path
                """,
                (doc_id, name, path, time.time()),
            )

    def replace_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        with self.connect() as conn:
            conn.execute("delete from chunks where doc_id = ?", (doc_id,))
            conn.executemany(
                """
                insert into chunks(
                  id, doc_id, text, page_start, page_end, parent_id, prev_id, next_id,
                  hierarchy_path, token_count, anchor_type, metadata
                )
                values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.parent_id,
                        chunk.prev_id,
                        chunk.next_id,
                        json.dumps(chunk.hierarchy_path, ensure_ascii=False),
                        chunk.token_count,
                        chunk.anchor_type,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )

    def clear_all(self) -> None:
        with self.connect() as conn:
            conn.execute("delete from chunks")
            conn.execute("delete from documents")

    def list_documents(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                select d.*,
                  (select count(*) from chunks c where c.doc_id = d.id) as chunk_count,
                  (select count(*) from chunks c where c.doc_id = d.id and c.anchor_type is not null) as anchor_count
                from documents d
                order by created_at desc
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self.connect() as conn:
            row = conn.execute("select * from chunks where id = ?", (chunk_id,)).fetchone()
            return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as conn:
            rows = conn.execute(f"select * from chunks where id in ({placeholders})", chunk_ids).fetchall()
        by_id = {row["id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[item] for item in chunk_ids if item in by_id]

    def all_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        query = "select * from chunks"
        params: tuple[str, ...] = ()
        if doc_id:
            query += " where doc_id = ?"
            params = (doc_id,)
        query += " order by rowid"
        with self.connect() as conn:
            return [self._row_to_chunk(row) for row in conn.execute(query, params).fetchall()]

    def anchor_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        query = "select * from chunks where anchor_type is not null"
        params: tuple[str, ...] = ()
        if doc_id:
            query += " and doc_id = ?"
            params = (doc_id,)
        query += " order by rowid"
        with self.connect() as conn:
            return [self._row_to_chunk(row) for row in conn.execute(query, params).fetchall()]

    def parent(self, chunk: Chunk) -> Chunk | None:
        return self.get_chunk(chunk.parent_id) if chunk.parent_id else None

    def siblings(self, chunk: Chunk) -> list[Chunk]:
        if not chunk.parent_id:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "select * from chunks where parent_id = ? and id != ? order by rowid",
                (chunk.parent_id, chunk.id),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def local_neighbors(self, chunk: Chunk, radius: int = 1) -> list[Chunk]:
        seen: set[str] = set()
        result: list[Chunk] = []

        def add(candidate_id: str | None) -> None:
            if not candidate_id or candidate_id in seen:
                return
            candidate = self.get_chunk(candidate_id)
            if candidate:
                seen.add(candidate.id)
                result.append(candidate)

        left = chunk.prev_id
        right = chunk.next_id
        for _ in range(radius):
            left_chunk = self.get_chunk(left) if left else None
            right_chunk = self.get_chunk(right) if right else None
            add(left)
            add(right)
            left = left_chunk.prev_id if left_chunk else None
            right = right_chunk.next_id if right_chunk else None
        return result

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            text=row["text"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            parent_id=row["parent_id"],
            prev_id=row["prev_id"],
            next_id=row["next_id"],
            hierarchy_path=json.loads(row["hierarchy_path"]),
            token_count=row["token_count"],
            anchor_type=row["anchor_type"],
            metadata=json.loads(row["metadata"]),
        )
