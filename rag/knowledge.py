"""Editable copy of the knowledge base: one SQLite row per Chroma chunk.

Chroma holds what retrieval actually searches, but it is a poor place to read
or tune content -- there is no way to browse it, and a chunk's text can only be
changed by re-ingesting the whole document it came from. This table is the
readable, editable side of the same data: `content` is exactly the text that is
embedded and shown to the model, and `metadata` is the chunk's Chroma metadata
as a JSON object (`source`, `heading`, `chunk_index`, plus anything added).

**Every write goes to Chroma too.** update/create/delete re-embed the chunk and
upsert (or delete) it in the collection before the row is committed, so an edit
changes what both reply paths retrieve -- `rag.retrieve()` and the n8n agent's
`academy_docs` tool read the same collection. If the embed or the Chroma write
fails, the row is left untouched rather than drifting from what is live.

**The .docx files win.** `rag.ingest` rewrites the rows for every file it
ingests (content, metadata, `edited` back to 0) and drops rows for sections the
file no longer has. Edits to ingested chunks are therefore for tuning between
document updates; a change that must survive a re-ingest belongs in the source
document -- or in a manual chunk (`origin='manual'`), which ingest never
touches.

**No vectors here.** They are derived from `content` and recomputed on every
write, so they can never disagree with the text they claim to embed.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sqlite3
import uuid
from datetime import datetime, timezone

# Same file as the thread store: one database to back up. `KNOWLEDGE_DB`
# overrides it, `THREADS_DB` is honoured so both tables still move together.
DB_PATH = pathlib.Path(
    os.environ.get(
        "KNOWLEDGE_DB",
        os.environ.get(
            "THREADS_DB",
            str(pathlib.Path(__file__).resolve().parent.parent / "data" / "threads.db"),
        ),
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    -- The Chroma id. `<file name>#<chunk index>` for ingested chunks,
    -- `manual:<hex>` for ones added by hand.
    id          TEXT PRIMARY KEY,
    -- Exactly what is embedded and handed to the model, heading trail included.
    content     TEXT NOT NULL,
    -- JSON object, flat: Chroma metadata values must be str/int/float/bool.
    metadata    TEXT NOT NULL DEFAULT '{}',
    origin      TEXT NOT NULL DEFAULT 'ingest' CHECK (origin IN ('ingest', 'manual')),
    -- 1 once changed through the API/CLI/UI; ingest resets it to 0.
    edited      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS knowledge_chunks_origin ON knowledge_chunks (origin);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def connect(path: pathlib.Path | None = None):
    """A connection with the schema applied. Commits on clean exit."""
    target = pathlib.Path(path) if path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "content": row["content"],
        "metadata": json.loads(row["metadata"] or "{}"),
        "origin": row["origin"],
        "edited": bool(row["edited"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _dump(metadata: dict) -> str:
    return json.dumps(metadata, ensure_ascii=False)


def validate_metadata(metadata: object, *, defaults: dict | None = None) -> dict:
    """A copy of `metadata` Chroma will accept, or ValueError saying why not.

    Chroma only stores flat scalar metadata, so nested objects, lists and nulls
    are rejected here with a readable message instead of surfacing later as a
    Chroma error. `source` and `heading` are what retrieval reports for a hit
    (see rag.core.retrieve), so they are filled from `defaults` when missing.
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    out = dict(defaults or {})
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            out[key] = value
        else:
            raise ValueError(
                f"metadata[{key!r}] must be a string, number or boolean "
                f"(got {type(value).__name__}); Chroma stores flat values only"
            )
    for key in ("source", "heading"):
        if not isinstance(out.get(key), str):
            raise ValueError(f"metadata[{key!r}] must be a string")
    return out


def _validate_content(content: object) -> str:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content cannot be empty")
    return content


# --- Chroma side ------------------------------------------------------------
# Imported lazily: rag.core imports this module inside ingest(), and the
# pure-table functions below must work without Chroma or the embedder.


def _push(chunks: list[tuple[str, str, dict]]) -> None:
    """Embed and upsert (id, content, metadata) triples. Raises on failure."""
    if not chunks:
        return
    from .core import _collection, embed

    ids, documents, metadatas = zip(*chunks)
    _collection(create=True).upsert(
        ids=list(ids),
        documents=list(documents),
        metadatas=list(metadatas),
        embeddings=embed(list(documents)),
    )


def _drop(ids: list[str]) -> None:
    if not ids:
        return
    from .core import _collection

    _collection(create=True).delete(ids=ids)


# --- Reads ------------------------------------------------------------------


def list_chunks(
    *, q: str = "", source: str = "", path: pathlib.Path | None = None
) -> list[dict]:
    """All chunks, ordered by source file then position. `q` is a
    case-insensitive substring match on content, id and metadata."""
    with connect(path) as conn:
        rows = [_row(r) for r in conn.execute("SELECT * FROM knowledge_chunks")]
    if source:
        rows = [r for r in rows if r["metadata"].get("source") == source]
    if q:
        needle = q.lower()
        rows = [
            r
            for r in rows
            if needle in r["content"].lower()
            or needle in r["id"].lower()
            or needle in _dump(r["metadata"]).lower()
        ]
    rows.sort(
        key=lambda r: (
            r["origin"] != "ingest",
            str(r["metadata"].get("source", "")),
            r["metadata"].get("chunk_index", 0) if isinstance(r["metadata"].get("chunk_index"), int) else 0,
            r["id"],
        )
    )
    return rows


def get_chunk(chunk_id: str, *, path: pathlib.Path | None = None) -> dict | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
    return _row(row) if row else None


def count(*, path: pathlib.Path | None = None) -> int:
    with connect(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]


# --- Edits (table + Chroma) ------------------------------------------------


def update_chunk(
    chunk_id: str,
    *,
    content: str | None = None,
    metadata: dict | None = None,
    path: pathlib.Path | None = None,
) -> dict:
    """Change a chunk's text and/or metadata, re-embed it, update Chroma.

    `metadata`, when given, replaces the whole object (after validation) --
    send the full object back, not a patch. KeyError if the id is unknown,
    ValueError for invalid input; embedding/Chroma errors propagate and leave
    the row unchanged.
    """
    current = get_chunk(chunk_id, path=path)
    if current is None:
        raise KeyError(chunk_id)
    new_content = _validate_content(content) if content is not None else current["content"]
    new_metadata = (
        validate_metadata(metadata) if metadata is not None else current["metadata"]
    )
    if new_content == current["content"] and new_metadata == current["metadata"]:
        return current

    _push([(chunk_id, new_content, new_metadata)])
    with connect(path) as conn:
        conn.execute(
            """UPDATE knowledge_chunks
                  SET content = ?, metadata = ?, edited = 1, updated_at = ?
                WHERE id = ?""",
            (new_content, _dump(new_metadata), _now(), chunk_id),
        )
    return get_chunk(chunk_id, path=path)


def create_chunk(
    content: str,
    metadata: dict | None = None,
    *,
    chunk_id: str | None = None,
    path: pathlib.Path | None = None,
) -> dict:
    """Add a hand-written chunk. Ingest never overwrites or removes these."""
    content = _validate_content(content)
    metadata = validate_metadata(
        metadata or {}, defaults={"source": "manual", "heading": "Manual entry"}
    )
    chunk_id = chunk_id or f"manual:{uuid.uuid4().hex[:12]}"
    if get_chunk(chunk_id, path=path) is not None:
        raise ValueError(f"a chunk with id {chunk_id!r} already exists")

    _push([(chunk_id, content, metadata)])
    now = _now()
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO knowledge_chunks
                   (id, content, metadata, origin, edited, created_at, updated_at)
               VALUES (?, ?, ?, 'manual', 1, ?, ?)""",
            (chunk_id, content, _dump(metadata), now, now),
        )
    return get_chunk(chunk_id, path=path)


def delete_chunk(chunk_id: str, *, path: pathlib.Path | None = None) -> None:
    """Remove a chunk from Chroma and the table. An ingested chunk comes back
    on the next ingest of its file (the .docx wins)."""
    if get_chunk(chunk_id, path=path) is None:
        raise KeyError(chunk_id)
    _drop([chunk_id])
    with connect(path) as conn:
        conn.execute("DELETE FROM knowledge_chunks WHERE id = ?", (chunk_id,))


# --- Bulk: export / import --------------------------------------------------


def export_chunks(*, path: pathlib.Path | None = None) -> list[dict]:
    return list_chunks(path=path)


def import_chunks(entries: list[dict], *, path: pathlib.Path | None = None) -> dict:
    """Apply an edited export. Returns {"updated", "created", "unchanged"} ids.

    Each entry needs `content` and `metadata`; `id` is optional. An entry whose
    id exists is updated if its content or metadata differ; one with an unknown
    or missing id is created as a manual chunk. Chunks absent from the file are
    left alone -- deleting is explicit (delete_chunk), never inferred from an
    omission. Everything is validated before anything is written, and the
    changed chunks are embedded in one batch.
    """
    if not isinstance(entries, list):
        raise ValueError("import file must be a JSON array of chunks")

    existing = {c["id"]: c for c in list_chunks(path=path)}
    updates: list[tuple[str, str, dict]] = []
    creates: list[tuple[str, str, dict]] = []
    unchanged: list[str] = []
    seen: set[str] = set()

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {i} is not an object")
        try:
            content = _validate_content(entry.get("content"))
            chunk_id = entry.get("id") or None
            current = existing.get(chunk_id) if chunk_id else None
            if current is None:
                metadata = validate_metadata(
                    entry.get("metadata") or {},
                    defaults={"source": "manual", "heading": "Manual entry"},
                )
            else:
                metadata = validate_metadata(entry.get("metadata", current["metadata"]))
        except ValueError as exc:
            raise ValueError(f"entry {i} ({entry.get('id') or 'new'}): {exc}") from exc

        chunk_id = chunk_id or f"manual:{uuid.uuid4().hex[:12]}"
        if chunk_id in seen:
            raise ValueError(f"entry {i}: duplicate id {chunk_id!r} in the file")
        seen.add(chunk_id)

        if current is None:
            creates.append((chunk_id, content, metadata))
        elif content != current["content"] or metadata != current["metadata"]:
            updates.append((chunk_id, content, metadata))
        else:
            unchanged.append(chunk_id)

    _push(updates + creates)
    now = _now()
    with connect(path) as conn:
        for chunk_id, content, metadata in updates:
            conn.execute(
                """UPDATE knowledge_chunks
                      SET content = ?, metadata = ?, edited = 1, updated_at = ?
                    WHERE id = ?""",
                (content, _dump(metadata), now, chunk_id),
            )
        for chunk_id, content, metadata in creates:
            conn.execute(
                """INSERT INTO knowledge_chunks
                       (id, content, metadata, origin, edited, created_at, updated_at)
                   VALUES (?, ?, ?, 'manual', 1, ?, ?)""",
                (chunk_id, content, _dump(metadata), now, now),
            )
    return {
        "updated": [c[0] for c in updates],
        "created": [c[0] for c in creates],
        "unchanged": unchanged,
    }


# --- Ingest hooks (called by rag.core.ingest) ------------------------------


def stale_ingest_ids(
    sources: list[str], keep: set[str], *, path: pathlib.Path | None = None
) -> list[str]:
    """Ingested chunk ids for these source files that the new ingest did not
    produce -- sections removed from the document since it was last ingested."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id, metadata FROM knowledge_chunks WHERE origin = 'ingest'"
        ).fetchall()
    wanted = set(sources)
    return [
        r["id"]
        for r in rows
        if r["id"] not in keep and json.loads(r["metadata"]).get("source") in wanted
    ]


def record_ingest(
    chunks: list[tuple[str, str, dict]],
    *,
    removed: list[str] = (),
    reset: bool = False,
    path: pathlib.Path | None = None,
) -> None:
    """Mirror an ingest into the table: the documents overwrite their rows.

    `reset` clears every ingested row first (the collection was dropped);
    manual rows always survive. `removed` are stale ids already deleted from
    Chroma by the caller.
    """
    now = _now()
    with connect(path) as conn:
        if reset:
            conn.execute("DELETE FROM knowledge_chunks WHERE origin = 'ingest'")
        conn.executemany(
            "DELETE FROM knowledge_chunks WHERE id = ?", [(i,) for i in removed]
        )
        conn.executemany(
            """INSERT INTO knowledge_chunks
                   (id, content, metadata, origin, edited, created_at, updated_at)
               VALUES (?, ?, ?, 'ingest', 0, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   content = excluded.content,
                   metadata = excluded.metadata,
                   origin = 'ingest',
                   edited = 0,
                   updated_at = excluded.updated_at""",
            [(i, c, _dump(m), now, now) for i, c, m in chunks],
        )


def manual_chunks(*, path: pathlib.Path | None = None) -> list[tuple[str, str, dict]]:
    """Manual rows as (id, content, metadata), for re-pushing after a reset."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM knowledge_chunks WHERE origin = 'manual'"
        ).fetchall()
    return [(r["id"], r["content"], json.loads(r["metadata"])) for r in rows]


# --- Backfill ---------------------------------------------------------------


def pull_from_chroma(*, path: pathlib.Path | None = None) -> int:
    """Copy the live collection into the table, without re-embedding.

    For a collection ingested before this table existed. Existing rows with
    the same id are overwritten with what Chroma currently holds. Returns the
    number of chunks copied.
    """
    from .core import _collection

    result = _collection(create=True).get(include=["documents", "metadatas"])
    now = _now()
    rows = []
    for chunk_id, document, metadata in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        metadata = dict(metadata or {})
        origin = "manual" if chunk_id.startswith("manual:") else "ingest"
        rows.append((chunk_id, document or "", _dump(metadata), origin, now, now))
    with connect(path) as conn:
        conn.executemany(
            """INSERT INTO knowledge_chunks
                   (id, content, metadata, origin, edited, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   content = excluded.content,
                   metadata = excluded.metadata,
                   updated_at = excluded.updated_at""",
            rows,
        )
    return len(rows)
