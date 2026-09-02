"""SQLite record of every Academy conversation, keyed by Outlook conversationId.

Phase 3. Phases 1 and 2 are stateless: an email arrives, it is classified, it is
answered from the documents, nothing is remembered. That breaks the moment
somebody replies to a draft -- "and what about the second level?" retrieves
nothing useful, because the question only means something next to the message
before it.

This module is the memory. It stores what arrived and what was drafted back,
and hands `threads.context` the history to put in front of the next question.

**One row per turn.** A conversation is a sequence of (query, reply) pairs, not
two independent streams stitched together by insertion order. `record_inbound`
opens a row with the reply columns empty; `record_reply` fills in the most
recent open row for that conversation. A row with `reply IS NULL` is a question
that has not been answered yet -- there is no other way to represent "asked but
not answered" once query and reply live in the same row, which is the point:
the old two-table-shaped-as-one design could drift (a reply recorded with
nothing to pair it to, an inbound counted as unanswered forever after its reply
silently failed to record) in a way this cannot.

Two deliberate choices carried over unchanged:

**Outlook's conversationId is the key.** It is assigned by the server, it
survives subject-line edits and "Re: Re: FW:" accretion, and it is already on
the trigger payload. Where one is genuinely absent -- a manual test, a transport
that has no such concept -- `subject_key()` derives a fallback, and it is a
fallback, not a peer: two enquirers who both write "Course dates?" collapse into
one thread under it.

**Inbound messages are deduplicated on internetMessageId.** An n8n trigger that
re-fires, or a workflow re-run against the same mailbox, otherwise opens the
same email as a new turn twice -- and the model then sees the enquirer asking
the same question twice. `record_inbound` returns False when it has seen the id
before, which is also the signal the API uses to avoid drafting a second reply
to one email.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

DB_PATH = pathlib.Path(
    os.environ.get(
        "THREADS_DB",
        str(pathlib.Path(__file__).resolve().parent.parent / "data" / "threads.db"),
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id     TEXT PRIMARY KEY,
    subject             TEXT NOT NULL DEFAULT '',
    -- Rolling summary of turns already dropped from the live window, and the
    -- exchange id it covers up to. See threads.context.
    summary             TEXT NOT NULL DEFAULT '',
    summarised_through  INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exchanges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL,
    query            TEXT NOT NULL,
    query_subject    TEXT NOT NULL DEFAULT '',
    message_id       TEXT,
    -- NULL until record_reply fills these in. A row with reply IS NULL is a
    -- question that has not been answered yet.
    reply            TEXT,
    reply_subject    TEXT,
    grounded         INTEGER,
    created_at       TEXT NOT NULL,
    replied_at       TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS exchanges_by_thread
    ON exchanges (conversation_id, id);

-- Partial, so the many rows with no message_id (drafts, manual tests) do not
-- collide on NULL.
CREATE UNIQUE INDEX IF NOT EXISTS exchanges_dedupe
    ON exchanges (message_id) WHERE message_id IS NOT NULL;
"""


@dataclass
class Exchange:
    """One query and, once answered, the reply drafted back to it."""

    id: int
    query: str
    query_subject: str = ""
    reply: str | None = None
    reply_subject: str | None = None
    grounded: bool | None = None
    created_at: str = ""
    replied_at: str | None = None

    @property
    def answered(self) -> bool:
        return self.reply is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def connect(path: pathlib.Path | None = None):
    """A connection with the schema applied. Commits on clean exit."""
    target = pathlib.Path(path) if path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL so a read during a write does not raise "database is locked" -- uvicorn
    # serves these handlers from a thread pool, so concurrent access is normal.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- Keys -------------------------------------------------------------------

_RE_PREFIX = re.compile(r"^\s*((re|fw|fwd|aw|antw|tr)\s*(\[\d+\])?\s*:\s*)+", re.I)


def subject_key(subject: str) -> str:
    """Fallback thread key derived from a subject line.

    Only for messages arriving without a conversationId. Strips any stack of
    reply/forward prefixes and folds case and whitespace, so "FW: Re: Course
    dates" and "course dates" land together.
    """
    stripped = _RE_PREFIX.sub("", subject or "").strip()
    folded = re.sub(r"\s+", " ", stripped).lower()
    return f"subject:{folded}" if folded else "subject:(none)"


def thread_key(conversation_id: str | None, subject: str = "") -> str:
    """The conversationId when there is one, else a subject-derived key."""
    if conversation_id and conversation_id.strip():
        return conversation_id.strip()
    return subject_key(subject)


# --- Writes -----------------------------------------------------------------


def ensure_conversation(
    conversation_id: str, subject: str = "", *, path: pathlib.Path | None = None
) -> None:
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO conversations
                   (conversation_id, subject, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(conversation_id) DO UPDATE SET
                   updated_at = excluded.updated_at,
                   -- Keep the first subject seen; later ones carry "Re:" noise.
                   subject = CASE WHEN conversations.subject = ''
                                  THEN excluded.subject
                                  ELSE conversations.subject END""",
            (conversation_id, subject or "", _now(), _now()),
        )


def record_inbound(
    conversation_id: str,
    body: str,
    *,
    subject: str = "",
    message_id: str | None = None,
    path: pathlib.Path | None = None,
) -> bool:
    """Open a new turn for an arriving email. False if message_id was already stored.

    The caller is expected to branch on the return value: a False means the
    trigger delivered something already handled, and drafting a second reply to
    it would put two drafts in the mailbox for one email.
    """
    ensure_conversation(conversation_id, subject, path=path)
    with connect(path) as conn:
        try:
            conn.execute(
                """INSERT INTO exchanges
                       (conversation_id, query, query_subject, message_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (conversation_id, body, subject or "", message_id, _now()),
            )
        except sqlite3.IntegrityError:
            return False  # exchanges_dedupe fired: same internetMessageId.
    return True


def record_reply(
    conversation_id: str,
    body: str,
    *,
    subject: str = "",
    grounded: bool = False,
    path: pathlib.Path | None = None,
) -> None:
    """Fill in the reply half of the most recently opened, still-unanswered turn.

    Drafted, not sent -- nothing here is ever sent. Pairs with the `record_inbound`
    call for the same conversation earlier in the same request; every caller in
    this codebase records the inbound and then, once it has an answer, records
    the reply before handling anything else for that conversation, so "the most
    recent unanswered row" is always the one this reply belongs to.
    """
    ensure_conversation(conversation_id, subject, path=path)
    with connect(path) as conn:
        conn.execute(
            """UPDATE exchanges
                  SET reply = ?, reply_subject = ?, grounded = ?, replied_at = ?
                WHERE id = (
                    SELECT id FROM exchanges
                     WHERE conversation_id = ? AND reply IS NULL
                     ORDER BY id DESC LIMIT 1
                )""",
            (body, subject or "", int(grounded), _now(), conversation_id),
        )


def set_summary(
    conversation_id: str,
    summary: str,
    through_id: int,
    *,
    path: pathlib.Path | None = None,
) -> None:
    """Record the rolling summary and how far into the thread it reaches."""
    with connect(path) as conn:
        conn.execute(
            """UPDATE conversations
                  SET summary = ?, summarised_through = ?, updated_at = ?
                WHERE conversation_id = ?""",
            (summary, through_id, _now(), conversation_id),
        )


# --- Reads ------------------------------------------------------------------


def history(
    conversation_id: str, *, limit: int = 0, path: pathlib.Path | None = None
) -> list[Exchange]:
    """Turns oldest-first. `limit` keeps the most recent N, still in order."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM exchanges WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()

    exchanges = [
        Exchange(
            id=row["id"],
            query=row["query"],
            query_subject=row["query_subject"],
            reply=row["reply"],
            reply_subject=row["reply_subject"],
            grounded=None if row["grounded"] is None else bool(row["grounded"]),
            created_at=row["created_at"],
            replied_at=row["replied_at"],
        )
        for row in rows
    ]
    return exchanges[-limit:] if limit else exchanges


def get_summary(
    conversation_id: str, *, path: pathlib.Path | None = None
) -> tuple[str, int]:
    """(summary, summarised_through). ("", 0) for an unknown conversation."""
    with connect(path) as conn:
        row = conn.execute(
            """SELECT summary, summarised_through FROM conversations
                WHERE conversation_id = ?""",
            (conversation_id,),
        ).fetchone()
    return ("", 0) if row is None else (row["summary"], row["summarised_through"])


def conversations(*, path: pathlib.Path | None = None) -> list[dict]:
    """Every thread with its turn count, most recently updated first."""
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT c.conversation_id, c.subject, c.updated_at,
                      COUNT(e.id) AS turns,
                      COALESCE(SUM(e.reply IS NOT NULL), 0) AS answered
                 FROM conversations c
                 LEFT JOIN exchanges e USING (conversation_id)
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC"""
        ).fetchall()
    return [dict(row) for row in rows]
