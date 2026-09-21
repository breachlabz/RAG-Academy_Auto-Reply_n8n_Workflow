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
    -- Graph message id of the inbound email (Prepare's `ref`). Carried on the
    -- row so the review queue can reply to the right message without n8n.
    ref              TEXT,
    -- NULL until record_reply fills these in. A row with reply IS NULL is a
    -- question that has not been answered yet.
    reply            TEXT,
    reply_subject    TEXT,
    grounded         INTEGER,
    created_at       TEXT NOT NULL,
    replied_at       TEXT,
    -- One-line gist of `query`, generated lazily the first time the review
    -- queue lists this row, then cached here so it is never regenerated.
    query_gist       TEXT,
    -- Set once a human has reviewed and sent the reply from the review queue.
    -- `edited_reply` is what was actually sent -- equal to `reply` when the
    -- human sent it unchanged, different when they edited it first.
    edited_reply     TEXT,
    sent             INTEGER NOT NULL DEFAULT 0,
    sent_at          TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS exchanges_by_thread
    ON exchanges (conversation_id, id);

-- Partial, so the many rows with no message_id (drafts, manual tests) do not
-- collide on NULL.
CREATE UNIQUE INDEX IF NOT EXISTS exchanges_dedupe
    ON exchanges (message_id) WHERE message_id IS NOT NULL;
"""

# Columns added after the original schema shipped. SQLite has no
# `ADD COLUMN IF NOT EXISTS`, so `connect()` checks `table_info` and adds
# whatever is missing -- lets an existing threads.db pick these up in place
# rather than needing a reset.
_ADDED_COLUMNS = {
    "ref": "TEXT",
    "query_gist": "TEXT",
    "edited_reply": "TEXT",
    "sent": "INTEGER NOT NULL DEFAULT 0",
    "sent_at": "TEXT",
    # n8n's `$execution.resumeUrl` for the Wait node paused on this reply --
    # POSTing here lets /review/{id}/send resume that execution once a human
    # approves. NULL for rows recorded before this existed, or by a caller
    # that never waits on review (e.g. the one-call /generate-reply path).
    "resume_url": "TEXT",
    # A scheduled follow-up on a thread, not a reply to a real inbound
    # question -- see create_followup(). `query` holds the topic (what to
    # draft about) rather than something the enquirer wrote, and `due_at` is
    # when a scheduler should draft and surface it. A row with is_followup=0
    # (every row before this existed) is an ordinary enquiry/reply turn.
    "is_followup": "INTEGER NOT NULL DEFAULT 0",
    "due_at": "TEXT",
}

# Same idea, for conversations. Normalised (lowercase, trimmed) at write time
# in ensure_conversation -- comparing addresses case- or whitespace-sensitively
# would silently fail to notice "John@X.com" and "john@x.com" are the same
# sender. Blank for a conversation whose inbound email carried no `from`
# (a manual test, a transport that has no such concept) -- sender_threads()
# below is a no-op for those, not an error.
_ADDED_COLUMNS_CONVERSATIONS = {
    "sender_email": "TEXT NOT NULL DEFAULT ''",
}


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(exchanges)")}
    for column, decl in _ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE exchanges ADD COLUMN {column} {decl}")

    existing_conv = {
        row["name"] for row in conn.execute("PRAGMA table_info(conversations)")
    }
    for column, decl in _ADDED_COLUMNS_CONVERSATIONS.items():
        if column not in existing_conv:
            conn.execute(f"ALTER TABLE conversations ADD COLUMN {column} {decl}")

    # Index creation deferred to here rather than the static SCHEMA string --
    # sender_email does not exist yet on a fresh database until the ALTER
    # TABLE above runs, so an index on it inside CREATE TABLE IF NOT EXISTS
    # would fail on first boot.
    conn.execute(
        """CREATE INDEX IF NOT EXISTS conversations_by_sender
               ON conversations (sender_email) WHERE sender_email != ''"""
    )


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
    # True for a scheduled follow-up (see create_followup): `query` is a
    # topic label the system chose, not something the enquirer wrote. See
    # threads.context.render, the one place this distinction actually
    # matters -- everywhere else an Exchange is just a row.
    is_followup: bool = False
    sent: bool = False

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
    _migrate(conn)
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


def normalise_email(address: str | None) -> str:
    """Lowercase and trimmed, so "John@X.com" and "john@x.com " compare equal.
    "" for anything blank -- never raises on a malformed or missing address."""
    return (address or "").strip().lower()


def ensure_conversation(
    conversation_id: str,
    subject: str = "",
    *,
    sender_email: str = "",
    path: pathlib.Path | None = None,
) -> None:
    sender_email = normalise_email(sender_email)
    with connect(path) as conn:
        conn.execute(
            """INSERT INTO conversations
                   (conversation_id, subject, sender_email, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(conversation_id) DO UPDATE SET
                   updated_at = excluded.updated_at,
                   -- Keep the first subject seen; later ones carry "Re:" noise.
                   subject = CASE WHEN conversations.subject = ''
                                  THEN excluded.subject
                                  ELSE conversations.subject END,
                   -- Same for sender_email -- fill it in if it was missing
                   -- (an older row, or a caller that did not have it yet),
                   -- never overwrite one already on record.
                   sender_email = CASE WHEN conversations.sender_email = ''
                                  THEN excluded.sender_email
                                  ELSE conversations.sender_email END""",
            (conversation_id, subject or "", sender_email, _now(), _now()),
        )


def record_inbound(
    conversation_id: str,
    body: str,
    *,
    subject: str = "",
    message_id: str | None = None,
    ref: str | None = None,
    sender_email: str = "",
    path: pathlib.Path | None = None,
) -> int | None:
    """Open a new turn for an arriving email. Returns the new row's id, or
    None if message_id was already stored.

    The caller is expected to branch on the return value: None means the
    trigger delivered something already handled, and drafting a second reply
    to it would put two drafts in the mailbox for one email. A real id
    should be threaded all the way through to record_reply's `exchange_id`
    -- see there for why guessing "the most recent unanswered turn" instead
    is not safe once more than one row on a thread can be unanswered at once
    (a near-duplicate suppressed by is_near_duplicate stays that way
    forever; a genuine race between two fast inbound emails is rarer but not
    impossible).

    `ref` is Graph's message id for this email (Prepare's `ref`), stored so the
    review queue can send a reply to the right message without going back
    through n8n. `sender_email` is who it was from -- lets sender_threads()
    below tell a reviewer this enquirer has other open threads.
    """
    ensure_conversation(conversation_id, subject, sender_email=sender_email, path=path)
    with connect(path) as conn:
        try:
            cur = conn.execute(
                """INSERT INTO exchanges
                       (conversation_id, query, query_subject, message_id, ref,
                        created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conversation_id, body, subject or "", message_id, ref, _now()),
            )
        except sqlite3.IntegrityError:
            return None  # exchanges_dedupe fired: same internetMessageId.
    return cur.lastrowid


def record_reply(
    conversation_id: str,
    body: str,
    *,
    subject: str = "",
    grounded: bool = False,
    resume_url: str | None = None,
    exchange_id: int | None = None,
    path: pathlib.Path | None = None,
) -> None:
    """Fill in the reply half of a turn.

    `exchange_id`, when given, targets that exact row -- always prefer this.
    The split Outlook path (Prepare -> Finalize -> /threads/reply) threads
    the id `record_inbound` returned all the way through for exactly this.

    Without it, falls back to guessing "the most recently opened, still-
    unanswered turn" -- kept only for the one-call /generate-reply path,
    where this runs synchronously right after `record_inbound` in the same
    request with nothing else able to create a newer unanswered row in
    between, so there is nothing for the guess to get wrong. On any path
    where the inbound and the reply are two separate requests, the guess is
    not safe: a near-duplicate suppressed by `is_near_duplicate` sits at
    `reply IS NULL` forever (see api.py), and if it is the most recent such
    row when an unrelated reply gets recorded, that reply lands on it
    instead of the turn it actually answers.

    `resume_url` is n8n's `$execution.resumeUrl` for the Wait node paused
    right after this reply was recorded, when the caller is that workflow.

    Excludes `is_followup` rows from the fallback guess either way: a
    scheduled follow-up (see `create_followup`) also sits at `reply IS
    NULL` until its own `draft_followup` fills it in, and is never what an
    ordinary reply is meant for.
    """
    ensure_conversation(conversation_id, subject, path=path)
    with connect(path) as conn:
        if exchange_id is not None:
            conn.execute(
                """UPDATE exchanges
                      SET reply = ?, reply_subject = ?, grounded = ?, replied_at = ?,
                          resume_url = ?
                    WHERE id = ? AND conversation_id = ? AND is_followup = 0""",
                (body, subject or "", int(grounded), _now(), resume_url, exchange_id, conversation_id),
            )
            return
        conn.execute(
            """UPDATE exchanges
                  SET reply = ?, reply_subject = ?, grounded = ?, replied_at = ?,
                      resume_url = ?
                WHERE id = (
                    SELECT id FROM exchanges
                     WHERE conversation_id = ? AND reply IS NULL AND is_followup = 0
                     ORDER BY id DESC LIMIT 1
                )""",
            (body, subject or "", int(grounded), _now(), resume_url, conversation_id),
        )


# --- Follow-ups ---------------------------------------------------------
#
# A follow-up is a turn with no real inbound question behind it -- the
# system owes the enquirer a second message that isn't contingent on them
# writing back first (see project discussion: "a reply that doesn't expect
# anything from the user as acknowledgement"). Modelled as an ordinary
# `exchanges` row rather than a separate table: `query` holds a short topic
# label instead of something the enquirer wrote, `is_followup=1` marks it,
# and `due_at` is when a scheduler should draft it. Once drafted it is
# indistinguishable from any other row to `pending_review()`/`sent_history()`
# -- same review queue, same Send path, same sent-only-after-a-real-send
# guarantee -- so nothing downstream needed to change for this to work.
#
# Each stage is independent: scheduling a second follow-up is just another
# call to `create_followup`, not a link in a chain. A stage that never gets
# reviewed cannot block or break a later one.


def create_followup(
    conversation_id: str,
    topic: str,
    due_at: str,
    *,
    subject: str = "",
    ref: str | None = None,
    path: pathlib.Path | None = None,
) -> int:
    """Schedule one follow-up stage on a conversation. Returns the new row's id.

    `ref` is the Graph message id Send will reply to; when not given
    explicitly it is looked up as the most recent `ref` seen anywhere on
    this conversation, so the follow-up still threads into the right Outlook
    conversation even though nothing arrived to carry a `ref` of its own.
    """
    ensure_conversation(conversation_id, subject, path=path)
    with connect(path) as conn:
        if ref is None:
            row = conn.execute(
                """SELECT ref FROM exchanges
                    WHERE conversation_id = ? AND ref IS NOT NULL
                    ORDER BY id DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            ref = row["ref"] if row else None
        cur = conn.execute(
            """INSERT INTO exchanges
                   (conversation_id, query, query_subject, ref, is_followup,
                    due_at, created_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (conversation_id, topic, subject or "", ref, due_at, _now()),
        )
        return cur.lastrowid


def due_followups(
    *, before: str | None = None, path: pathlib.Path | None = None
) -> list[dict]:
    """Scheduled follow-ups whose due_at has passed and are not drafted yet,
    earliest-due first. `before` overrides "now" for testing."""
    cutoff = before or _now()
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT * FROM exchanges
                WHERE is_followup = 1 AND reply IS NULL AND due_at <= ?
                ORDER BY due_at""",
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def draft_followup(
    exchange_id: int,
    reply: str,
    *,
    grounded: bool = True,
    path: pathlib.Path | None = None,
) -> None:
    """Fill in one scheduled follow-up's reply, by exact row id.

    Unlike `record_reply`'s "most recently opened, still-unanswered turn"
    heuristic, a follow-up is targeted by id -- a conversation can have
    several independent follow-ups pending (or a follow-up pending
    alongside a genuinely unanswered inbound turn) and each has to land on
    the row it was actually drafted for, not whichever is most recent.
    """
    with connect(path) as conn:
        conn.execute(
            """UPDATE exchanges
                  SET reply = ?, grounded = ?, replied_at = ?
                WHERE id = ? AND is_followup = 1""",
            (reply, int(grounded), _now(), exchange_id),
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
            is_followup=bool(row["is_followup"]),
            sent=bool(row["sent"]),
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


def sender_threads(
    sender_email: str,
    *,
    exclude_conversation_id: str | None = None,
    path: pathlib.Path | None = None,
) -> list[dict]:
    """Other conversations from the same sender with something still open --
    an unanswered question, or a grounded reply still awaiting review.

    A visibility signal for a reviewer ("this enquirer has N other open
    threads"), not something this codebase acts on by itself: a shared inbox
    alias can have several different real people behind one address, so
    whether that matters is a judgment call for a human, never an automatic
    merge. "" for `sender_email` (no address captured on this conversation)
    always returns nothing -- two blank addresses matching each other would
    be a false link between enquirers who were never actually the same
    person, not a real one.
    """
    sender_email = normalise_email(sender_email)
    if not sender_email:
        return []
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT DISTINCT c.conversation_id, c.subject
                 FROM conversations c
                 JOIN exchanges e USING (conversation_id)
                WHERE c.sender_email = ?
                  AND c.conversation_id != ?
                  AND (
                        (e.reply IS NULL AND e.is_followup = 0)
                     OR (e.reply IS NOT NULL AND e.grounded = 1 AND e.sent = 0)
                      )
                ORDER BY c.updated_at DESC""",
            (sender_email, exclude_conversation_id or ""),
        ).fetchall()
    return [dict(row) for row in rows]


def sender_sent_threads(
    sender_email: str,
    *,
    exclude_conversation_id: str | None = None,
    limit: int = 20,
    path: pathlib.Path | None = None,
) -> list[dict]:
    """Other conversations from the same sender that have at least one sent
    reply -- the History-section counterpart to sender_threads() above.

    Kept as a separate query rather than folding into one "all other
    threads from this sender" call: sender_threads() answers "does this
    person have something outstanding elsewhere" (used where open work
    matters, the pending queue), this answers "what have we already sent
    this person" (used in History, where everything shown is already
    resolved and sender_threads() would always come back empty -- a fully
    sent thread never counts as "open"). Every caller already knows which
    question it's asking, so conflating them would only cost clarity.

    Same identity caveat as sender_threads(): a visibility signal for a
    human to read, never something this codebase merges or acts on by
    itself. Same blank-address guard too.
    """
    sender_email = normalise_email(sender_email)
    if not sender_email:
        return []
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT DISTINCT c.conversation_id, c.subject
                 FROM conversations c
                 JOIN exchanges e USING (conversation_id)
                WHERE c.sender_email = ?
                  AND c.conversation_id != ?
                  AND e.sent = 1
                ORDER BY c.updated_at DESC
                LIMIT ?""",
            (sender_email, exclude_conversation_id or "", limit),
        ).fetchall()
    return [dict(row) for row in rows]


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


# --- Review queue -------------------------------------------------------
#
# A grounded exchange (an AI-drafted reply that passed the grounding net) sits
# here, unsent, until a human reviews it in the review UI and sends it.
# `grounded=1 AND reply IS NOT NULL AND sent=0` is the queue -- no separate
# status column, because those three fields already say everything: not yet
# answered, answered but not grounded (never enters review), or grounded and
# waiting.


def pending_review(*, path: pathlib.Path | None = None) -> list[dict]:
    """Grounded replies not yet sent, oldest first (first drafted, first reviewed)."""
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT e.*, c.subject AS conversation_subject, c.sender_email AS conversation_sender_email
                 FROM exchanges e
                 JOIN conversations c USING (conversation_id)
                WHERE e.grounded = 1 AND e.reply IS NOT NULL AND e.sent = 0
                ORDER BY e.id"""
        ).fetchall()
    return [dict(row) for row in rows]


def sent_history(
    *, limit: int = 50, path: pathlib.Path | None = None
) -> list[dict]:
    """Already-sent replies, most recently sent first. For the review page's
    History section -- a record of what actually went out and whether it was
    edited before sending (`edited_reply` vs the original `reply`)."""
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT e.*, c.subject AS conversation_subject, c.sender_email AS conversation_sender_email
                 FROM exchanges e
                 JOIN conversations c USING (conversation_id)
                WHERE e.sent = 1
                ORDER BY e.sent_at DESC
                LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_exchange(exchange_id: int, *, path: pathlib.Path | None = None) -> dict | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM exchanges WHERE id = ?", (exchange_id,)
        ).fetchone()
    return dict(row) if row else None


def set_query_gist(
    exchange_id: int, gist: str, *, path: pathlib.Path | None = None
) -> None:
    """Cache the one-line query summary so it is generated at most once."""
    with connect(path) as conn:
        conn.execute(
            "UPDATE exchanges SET query_gist = ? WHERE id = ?", (gist, exchange_id)
        )


def mark_sent(
    exchange_id: int, edited_reply: str, *, path: pathlib.Path | None = None
) -> dict | None:
    """Record the human-approved final text and take the row out of the queue.

    Only fires on a row that is still actually pending -- `sent = 0` in the
    WHERE clause makes this idempotent (a retried request cannot flip an
    already-sent row) and lets the caller tell "sent" from "already sent" by
    checking whether a row was updated. Returns the row as it stood *before*
    this call so the caller (about to place the real Graph send) still has
    `ref` and `conversation_id` even though the row is now marked sent.
    """
    row = get_exchange(exchange_id, path=path)
    if row is None or row["sent"]:
        return None
    with connect(path) as conn:
        conn.execute(
            """UPDATE exchanges
                  SET edited_reply = ?, sent = 1, sent_at = ?
                WHERE id = ? AND sent = 0""",
            (edited_reply, _now(), exchange_id),
        )
    return row
