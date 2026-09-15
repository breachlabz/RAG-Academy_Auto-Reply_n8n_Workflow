"""Retrieval-augmented answering over the training documentation.

Phase 2 of the pipeline. `classifier.route()` decides an email *may* be
auto-answered; this module answers it -- but only out of text it actually
retrieved. The same fail-safe stance applies: if retrieval comes back weak, or
the model says the documents do not cover the question, `answer()` returns
`grounded=False` and the email belongs to a human. An ungrounded answer from a
9B is exactly the failure the Phase 1 gate exists to prevent, so it is not
allowed to leak back in here.

Embedding is bge-m3 (1024-dim) on its own llama-server, reached through the
same LiteLLM proxy as the chat models. Chroma is never given an embedding
function: every vector is computed here and passed in explicitly. That keeps
the embedder a property of this code rather than of the collection, so changing
it is a config change plus a re-ingest, not a Chroma migration.

Vectors from two different embedders are not comparable. Changing
`RAG_EMBED_MODEL` means `ingest --reset`, never a plain re-ingest -- the
dimensions differ (MiniLM's 384 against bge-m3's 1024), so a mixed collection
does not silently degrade, it fails outright.
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass, field

import chromadb
import requests

# Importing classifier.core also loads .env -- both halves of the pipeline read
# the same LiteLLM endpoint and key, so there is one place to change it.
from classifier.core import BASE_URL, NUM_CTX, SMALL_MODEL, auth_headers

# .md .txt .docx .pdf .html .htm. loaders.py imports nothing heavy at module
# level -- python-docx, pypdf and bs4 are all imported inside their loader
# function, so the API container (which queries but never ingests) does not need
# them installed to boot.
from .format_hint import LIST_DIRECTIVE, PROSE_DIRECTIVE, as_bullets
from .loaders import SUFFIXES

CHROMA_HOST = os.environ.get("RAG_CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("RAG_CHROMA_PORT", "8010"))
COLLECTION = os.environ.get("RAG_COLLECTION", "docs")

_DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = _DATA / "docs"

# Where new source documents land before ingest. Both directories are scanned
# by default_docs(); raw/ is the drop folder named in the project spec, docs/ is
# where the original corpus already lives, and there was no reason to move it.
RAW_DIR = _DATA / "raw"

# Cosine distance, so 0 is identical and 2 is opposite. Anything past this is
# treated as "the documents do not cover this" rather than fed to the model.
# The value is a starting guess, not a measurement -- sweep it against real
# questions the way EC_THRESHOLD was swept, or it is just a number.
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "1.0"))

TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

# Served by llama-bge-m3 with --embedding, routed through LiteLLM like
# everything else. bge-m3 takes raw text on both sides: it has no "query:" /
# "passage:" convention, and adding one -- as e5 and bge-v1.5 require -- would
# quietly shift every vector away from where it should sit.
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-m3")

# The embedder need not sit behind the same endpoint as the chat models. In the
# self-contained deployment the chat model is a shared LiteLLM elsewhere on the
# network while bge-m3 runs as a container in this compose, so RAG_EMBED_BASE_URL
# points at that local service. Unset, it falls back to the chat endpoint -- the
# original single-proxy setup, unchanged. RAG_EMBED_API_KEY likewise: only sent
# when set, so a keyless local embedder needs no dummy token.
EMBED_BASE_URL = os.environ.get("RAG_EMBED_BASE_URL", BASE_URL).rstrip("/")
EMBED_API_KEY = os.environ.get("RAG_EMBED_API_KEY", "")


def _embed_headers() -> dict[str, str]:
    """Auth for the embeddings call. An explicit RAG_EMBED_API_KEY wins; failing
    that, reuse the chat key only when the embedder shares the chat endpoint."""
    if EMBED_API_KEY:
        return {"Authorization": f"Bearer {EMBED_API_KEY}"}
    return auth_headers() if EMBED_BASE_URL == BASE_URL.rstrip("/") else {}

# The server is configured with --ubatch-size 8192, which bounds a single
# sequence, not a request. Batching keeps ingest to a few round trips while
# staying well inside it.
EMBED_BATCH = 16

# Chunks carry their heading trail into the embedding, so a row reading
# "Module: CAN Communication ..." still matches "does level 2 cover CAN" even
# though the body never mentions which level it belongs to.
#
# 3000 rather than 1200 so a whole section survives as one chunk. At 1200 the
# "Who can take the EVH training?" section split into three, one per level, and
# "what are the three levels?" then retrieved Level 1 and Level 3 while ranking
# Level 2 seventh -- the answer named two levels and reported the third as
# missing. A question about a section wants the section, not a third of it.
# Affordable because the corpus is small; revisit when it is not.
#
# Now env-tunable, but the default is a measured result and not a guess like
# MAX_DISTANCE is. Lower it and you are re-opening the split-section failure
# above; re-run scripts/rag_eval.py if you do.
MAX_CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", "3000"))

# Characters repeated from the end of one sub-chunk at the start of the next.
#
# Default 0, deliberately. Overlap exists to stop a fact being severed by an
# arbitrary cut, but this chunker only ever cuts at section headings and, within
# an over-long section, at blank lines -- both places where a sentence is
# already whole. Overlap there buys duplicate text in the context window and
# two near-identical hits eating two of the k=4 slots. It earns its keep once
# RAG_CHUNK_CHARS is small enough that sections routinely split; set it then.
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "0"))

# The refusal token. The model is told to emit this exact string when the
# context does not answer the question; seeing it flips grounded to False.
NO_ANSWER = "NOT_IN_DOCUMENTS"

# What to say when the documents do not cover the question. Deliberately not
# generated by the model: the one thing known for certain in this branch is
# that there was nothing to ground an answer in, so asking a 9B to phrase the
# apology just reopens the door to it filling the gap from memory.
#
# The second sentence is a promise, and it is kept -- every ungrounded email
# still goes to the human queue. If that ever stops being true, this text has
# to change with it.
NO_INFO_REPLY = os.environ.get(
    "RAG_NO_INFO_REPLY",
    "We don't have relevant information related to this query. "
    "A member of our team will follow up with you directly.",
)

# --- Email draft shell ------------------------------------------------------
#
# reply is a *draft* -- greeting, the grounded answer, sign-off -- ready to
# send. The shell is assembled deterministically here and is NOT generated by
# the model. The model's job is the factual body only, which retrieval has
# grounded; letting it write the greeting and sign-off too would reopen the gap
# it has been stopped from filling ("we look forward to seeing you at our
# campus" -- a fact nothing in the documents supports). Courtesy is a template;
# facts stay grounded.
#
# All four pieces are env-overridable so the wording and the sender travel with
# the deployment, not the code. \n in an env value is honoured.
def _env_text(name: str, default: str) -> str:
    return os.environ.get(name, default).replace("\\n", "\n")


EMAIL_SUBJECT = _env_text("RAG_EMAIL_SUBJECT", "Re: your enquiry")
EMAIL_GREETING = _env_text("RAG_EMAIL_GREETING", "Hello,")
EMAIL_SIGNOFF = _env_text("RAG_EMAIL_SIGNOFF", "Best regards,\nThe Training Team")

# Closing line differs by outcome: an answered enquiry invites a reply, an
# unanswered one is already promised a human, so it must not also say "just
# reply" as if the loop were closed.
EMAIL_CLOSER_ANSWERED = _env_text(
    "RAG_EMAIL_CLOSER_ANSWERED",
    "If you have any further questions, just reply to this email.",
)
EMAIL_CLOSER_NO_INFO = _env_text(
    "RAG_EMAIL_CLOSER_NO_INFO",
    "Thank you for your patience.",
)


def format_email(body: str, *, answered: bool) -> str:
    """Wrap a factual body in the greeting/sign-off shell. Body used verbatim."""
    closer = EMAIL_CLOSER_ANSWERED if answered else EMAIL_CLOSER_NO_INFO
    return "\n\n".join([EMAIL_GREETING, body.strip(), closer, EMAIL_SIGNOFF])

# Second net under the refusal token, because the token alone leaks. Measured
# on seven questions the document cannot answer, the 9B emitted NOT_IN_DOCUMENTS
# for four and answered the other three in prose -- "The provided documentation
# does not specify the number of days..." -- which arrives as a perfectly
# well-formed answer that happens to say nothing. With cosine distance unable
# to separate answerable from unanswerable here, the token was the only gate
# holding, so it needs backup.
#
# The test is deliberately narrow: a sentence that negates *the source*, not
# one that merely contains a negation. "Level 1 does not include hands-on
# exercises" is a real answer about content and must survive; "the provided
# text does not detail Level 2" is a refusal wearing an answer's clothes. Both
# a source word and a negated reporting verb have to appear in the same
# sentence.
_SOURCE_WORD = re.compile(
    r"\b(document|documentation|text|extract|context|material|information|"
    r"provided|given|above)\w*\b",
    re.I,
)
_SOURCE_NEGATION = re.compile(
    r"\b(?:no|not|never|cannot|can't|doesn't|don't|isn't|aren't|lacks?)\b"
    r"[^.]{0,40}?"
    r"\b(specif\w+|mention\w*|contain\w*|provide[sd]?|includ\w+|state[sd]?|"
    r"detail\w*|cover\w*|indicat\w*|describ\w*|list\w*|address\w*)\b",
    re.I,
)

# Splits the opening sentence at a leading contrastive conjunction. "X, but the
# documentation does not specify Y" is a real answer (X) with an honest caveat
# welded on, not a refusal -- but checking the whole first sentence catches the
# caveat's clause and discards X along with it. Only the clause before the
# conjunction is what "the first thing it says" actually means.
_CONTRAST_SPLIT = re.compile(r",?\s+\b(?:but|however|although|though|yet)\b,?\s*", re.I)


def refuses_in_prose(text: str) -> bool:
    """True when an answer OPENS with a refusal phrased as prose.

    Only the opening clause is checked. The system prompt's fallback for "the
    documents do not cover this" is the bare NOT_IN_DOCUMENTS token, so prose
    reaching for the same meaning states it as the first thing it says -- a
    refusal does not need three sentences of grounded content in front of it.
    Checking every sentence used to mean a trailing, honest caveat after a
    substantive answer ("...the documentation does not specify whether handouts
    are allowed") discarded the whole reply as a refusal, throwing away two
    correct sentences over one uncovered detail. Checking the whole first
    sentence has the same failure in miniature: "X, but the documentation does
    not specify Y" leads with real content and only pivots to the caveat after
    a conjunction, so the opening clause -- up to that conjunction, if any --
    is what gets tested, not the sentence as a whole.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences or not sentences[0]:
        return False
    first = sentences[0]
    contrast = _CONTRAST_SPLIT.search(first)
    opening = first[: contrast.start()] if contrast else first
    return bool(_SOURCE_WORD.search(opening) and _SOURCE_NEGATION.search(opening))


# Floor on what a stripped remainder has to be, in `_strip_leading_refusal`,
# to count as a real answer rather than a scrap not worth keeping. Not a
# business rule -- just cheap insurance against keeping something like "OK."
_MIN_SALVAGE_CHARS = 20


def _strip_leading_refusal(text: str) -> str:
    """Drop a leading sentence that opens with a refusal, keeping whatever
    substantive content follows -- the mirror image of the trailing-token
    strip in `ground()`, for the related-topic-answer shape the SYSTEM prompt
    now asks for: a small model given "lead with the fact, caveat after"
    reliably ignores the ordering and still opens with "The documentation
    does not specify X for Level 3." even when the rest of the same reply is
    a genuine, specific answer about a related level. That opening sentence
    is exactly what `refuses_in_prose` is built to catch -- discarding the
    whole reply over it throws away real content the same way the trailing
    NOT_IN_DOCUMENTS case used to.

    Iterates rather than stripping once: a model that leads with the refusal
    sometimes restates it in a second sentence before actually answering.
    Bounded automatically -- the SYSTEM prompt caps replies at a handful of
    sentences, and each iteration must find `refuses_in_prose` still true or
    it stops.
    """
    while text and refuses_in_prose(text):
        sentences = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        if len(sentences) < 2:
            return ""  # The whole (single-sentence) reply was the refusal.
        remainder = sentences[1].strip()
        if len(remainder) < _MIN_SALVAGE_CHARS:
            return ""
        text = remainder
    return text


def ground(text: str) -> str:
    """Clean a model's answer and decide what, if anything, is grounded.

    Returns the text to use as the reply, or "" when nothing here is grounded.
    Three shapes all collapse to "": an empty reply, the reply being exactly
    the bare NOT_IN_DOCUMENTS token as instructed, and a reply that is
    *entirely* a refusal phrased as prose (see `refuses_in_prose`) with
    nothing substantive following it.

    A fourth shape is not a refusal, and used to be treated as one: a model
    facing a multi-part question that the documents mostly cover sometimes
    answers what they do cover and appends the bare token for the part they
    don't, instead of replacing the whole reply with it as instructed. That is
    a partial answer, not nothing, so a *trailing* token is stripped and the
    grounded part in front of it is kept. A token that is not bare and not
    trailing -- buried mid-reply -- is not this shape, and is safer to discard
    whole than to guess which half is trustworthy.

    A fifth shape is the mirror of the fourth: the SYSTEM prompt now asks for
    a related-topic answer to lead with the fact and put any caveat after it,
    but a small model often ignores that and opens with the refusal sentence
    anyway even though real, on-topic content follows. `_strip_leading_refusal`
    drops that opening sentence (or sentences) and keeps the rest, the same
    salvage judgment as the trailing case, just at the other end of the reply.
    """
    text = text.strip()
    if not text or text == NO_ANSWER:
        return ""
    if text.endswith(NO_ANSWER):
        text = text[: -len(NO_ANSWER)].rstrip()
        if not text:
            return ""
    if NO_ANSWER in text:
        return ""
    return _strip_leading_refusal(text)

SYSTEM = f"""You answer enquiries about the training programmes described in
the documentation extracts provided.

Rules:
- Use only the extracts. Never use outside knowledge, and never guess a date,
  a price, a duration or a prerequisite that is not written in them. This
  applies especially to subjects you know about independently -- answering an
  automotive security question from your own knowledge rather than from the
  extracts is the failure this system exists to prevent. This rule applies no
  matter which of the three cases below you are in -- a related-topic answer
  still only ever states what the extracts actually say.
- If the extracts answer the question directly, answer from them as below.
- Before reaching for {NO_ANSWER}, check every extract for material on a
  related topic even if it does not name what was asked -- the same
  programme at a different level, an adjacent module, a general policy that
  most likely also applies. This is not a rare edge case: it is the normal
  outcome whenever the enquirer asks about specifics for one level and the
  extracts only cover a neighbouring one. Example: asked about the hardware
  kit for Level 3, and the extracts only describe the Level 2 kit -- that
  is related material, not nothing. Answer from it instead of refusing.
  Your first sentence must be a concrete fact from the extracts -- never
  "the documentation/extracts do not [specify/cover/mention/...]", not even
  as the opening clause of a longer sentence. State what the related material
  says first; only after that, in a separate trailing sentence, note the
  mismatch (for example: "...and you keep the kit after training. That is
  what applies to the Level 2 kit specifically -- the extracts do not say
  whether Level 3 uses the same one."). A reply that leads with what the
  documentation does not say is read as a refusal downstream and discarded
  outright, even when real content follows it -- so the opening sentence is
  not a style choice, it decides whether this answer reaches the enquirer at
  all.
- Only when nothing in the extracts is meaningfully related to the question
  either, reply with exactly {NO_ANSWER} and nothing else.
- Quote the specific module names and details from the extracts rather than
  paraphrasing them vaguely.
- Keep the answer short and addressed to the enquirer: at most four sentences
  of plain prose, OR a list of at most six short points.
- Use a bulleted list (one "- " item per line) when the enquirer wrote their
  query in points, asked for a list, numbered their questions, or asked
  something whose answer is naturally a set of items -- steps, options,
  prerequisites, dates, module names. Otherwise answer in plain prose. At most
  one short lead-in line before the list, and no section headings.
- Do not mention "extracts", "context" or "documents" in your answer."""

# Used instead of SYSTEM when a thread history is supplied. The two extra rules
# close the hole history opens: a model shown a previous reply will happily
# treat it as a source and re-state it as fact, which launders anything that
# slipped through earlier into permanent thread truth. The refusal rule has to
# be restated here too -- given a conversation to draw on, the model gets
# noticeably more willing to answer from it rather than emit the token.
THREAD_SYSTEM = f"""{SYSTEM}

Because you are shown the earlier conversation:
- The conversation tells you what was asked and how it was phrased. It is NOT a
  source of facts. Every factual claim -- including in a related-topic answer --
  must come from the extracts above, even if an earlier reply in the
  conversation stated it.
- If the extracts do not answer the latest question and contain nothing
  meaningfully related either, reply with exactly {NO_ANSWER}, even when the
  conversation appears to contain the answer."""


@dataclass
class Chunk:
    """One retrieved passage. `distance` is cosine, lower is closer."""

    text: str
    source: str
    heading: str
    distance: float


@dataclass
class Answer:
    """An answering outcome. Never raised -- failures arrive as `error`.

    `grounded` is the field to branch on: it is True only when retrieval was
    close enough, the model produced prose, and it did not refuse.
    """

    text: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    grounded: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def sources(self) -> list[str]:
        """Deduplicated headings behind the answer, in retrieval order."""
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.heading not in seen:
                seen.append(chunk.heading)
        return seen

    @property
    def reply(self) -> str:
        """A ready-to-send email draft: greeting, the answer, sign-off.

        Separate from `text` on purpose. `text` is the bare grounded body --
        empty unless an answer was actually grounded -- so a caller branching
        on it cannot mistake the fallback for a real answer, and `grounded`
        stays the single field that decides routing. `reply` is what you send;
        `text` is what you log.

        Empty on error. A failed request means it is not known whether the
        documents cover the question, and drafting "we don't have relevant
        information" would be a claim about the corpus that nothing supports.
        """
        if self.grounded:
            return format_email(self.text, answered=True)
        if self.error:
            return ""
        return format_email(NO_INFO_REPLY, answered=False)

    @property
    def subject(self) -> str:
        """Subject line for the draft. Empty whenever `reply` is."""
        return "" if (not self.grounded and self.error) else EMAIL_SUBJECT


def embed(texts: list[str], *, timeout: int = 180) -> list[list[float]]:
    """Embed texts through the proxy. Raises on failure; callers catch."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        resp = requests.post(
            f"{EMBED_BASE_URL}/embeddings",
            json={"model": EMBED_MODEL, "input": texts[start : start + EMBED_BATCH]},
            headers=_embed_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        # Sorted by index rather than trusting arrival order: the response is a
        # list of objects each carrying its own index, and silently misaligning
        # vectors with their chunks would poison the whole store invisibly.
        batch = sorted(resp.json()["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in batch)
    return vectors


def _client() -> chromadb.api.ClientAPI:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _collection(*, create: bool = False):
    """The collection, with no embedding function attached.

    embedding_function=None is deliberate. Chroma otherwise stores its default
    MiniLM in the collection config and reconstructs it on every get, which
    downloads an 80MB ONNX model into a container that never needs it -- and
    would silently embed with the wrong model if a caller ever passed
    query_texts instead of query_embeddings.
    """
    client = _client()
    if create:
        return client.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    return client.get_collection(COLLECTION, embedding_function=None)


def chunk_markdown(text: str, source: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_trail, body) pairs at h2/h3 boundaries.

    The heading trail is prepended to the stored text so the embedding sees
    what the section is about, not just its prose.
    """
    chunks: list[tuple[str, str]] = []
    trail: dict[int, str] = {}
    heading = source
    body: list[str] = []

    def flush() -> None:
        joined = "\n".join(body).strip()
        body.clear()
        if not joined:
            return
        # Long sections split on blank lines rather than mid-sentence.
        parts, current = [], ""
        for para in joined.split("\n\n"):
            if current and len(current) + len(para) + 2 > MAX_CHUNK_CHARS:
                parts.append(current)
                # Carry the tail of the part just closed into the next one, cut
                # back to a whitespace boundary so the overlap is whole words.
                # Only applies within an over-long section: consecutive sections
                # never overlap, because a heading is a real topic boundary and
                # bleeding one section's text into the next would put the wrong
                # heading trail on it.
                tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
                if tail and (space := tail.find(" ")) != -1:
                    tail = tail[space + 1 :]
                current = f"{tail}\n\n{para}" if tail else para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            parts.append(current)
        chunks.extend((heading, part) for part in parts)

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not match:
            body.append(line)
            continue
        level, title = len(match.group(1)), match.group(2).strip()
        if level >= 2:
            flush()
        trail[level] = title
        for deeper in [k for k in trail if k > level]:
            del trail[deeper]
        heading = " > ".join(trail[k] for k in sorted(trail))

    flush()
    return chunks


def load_text(path: pathlib.Path) -> str:
    """Read a document as markdown, converting if it is not already text."""
    from .loaders import load_text as _load

    return _load(path)


def ingest(paths: list[pathlib.Path], *, reset: bool = False) -> int:
    """Chunk and store the given markdown/text files. Returns chunks written.

    Ids are derived from the file name and chunk index, so re-ingesting an
    edited file overwrites its chunks instead of duplicating them. Sections
    *deleted* from a file still leave their old chunks behind, though -- use
    reset=True after removing content.
    """
    if reset:
        try:
            _client().delete_collection(COLLECTION)
        except Exception:
            pass  # Nothing to delete on a first run.

    collection = _collection(create=True)
    ids, documents, metadatas = [], [], []

    for path in paths:
        text = load_text(path)
        for index, (heading, part) in enumerate(chunk_markdown(text, path.stem)):
            ids.append(f"{path.name}#{index}")
            documents.append(f"{heading}\n\n{part}")
            # chunk_index duplicates what the id already encodes, but the id is
            # not returned by a query -- only metadata is, so anything a caller
            # needs to see about a hit has to live here.
            metadatas.append(
                {"source": path.name, "heading": heading, "chunk_index": index}
            )

    if not ids:
        return 0

    # Vectors are computed here and handed to Chroma, so the store never needs
    # to know which embedder produced them.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embed(documents),
    )
    return len(ids)


def retrieve(question: str, *, k: int = TOP_K) -> list[Chunk]:
    """Nearest chunks to the question, closest first, unfiltered."""
    result = _collection().query(query_embeddings=embed([question]), n_results=k)
    out = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        out.append(
            Chunk(
                text=text,
                source=str(meta.get("source", "")),
                heading=str(meta.get("heading", "")),
                distance=float(distance),
            )
        )
    return out


def answer(
    question: str,
    *,
    model: str = SMALL_MODEL,
    k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
    timeout: int = 120,
    history: str = "",
    query: str | None = None,
    as_list: bool | None = None,
) -> Answer:
    """Answer from the document set. Never raises -- see Answer.error.

    `history` is a rendered thread block (see `threads.context.load`). It is put
    in front of the model for phrasing and reference resolution only -- the
    grounding rules in SYSTEM still bind the *facts* to the retrieved extracts,
    because a previous reply in the thread is not a source. Without that
    distinction a single ungrounded sentence would become permanent thread
    context and every later answer could cite it.

    `query` overrides what gets embedded, for when the literal email text is a
    poor retrieval key ("and the second one?"). Defaults to `question`.

    `as_list` forces the reply shape: True adds an explicit "reply as a bulleted
    list" instruction and coerces the result into one if the model ignores it;
    False forces prose; None leaves it to the SYSTEM rule (which the small model
    reads as "prose" most of the time). Callers decide this from the enquirer's
    text -- see `rag.format_hint`.
    """
    try:
        chunks = retrieve(query or question, k=k)
    except Exception as exc:
        return Answer(error=f"retrieval failed: {exc}")

    if not chunks:
        return Answer(text="", chunks=[], grounded=False)

    # Drop chunks the index thinks are unrelated. If the closest one is already
    # past the cutoff there is nothing to answer from, and asking anyway just
    # invites the model to fill the gap from memory.
    kept = [chunk for chunk in chunks if chunk.distance <= max_distance]
    if not kept:
        return Answer(chunks=chunks, grounded=False)

    context = "\n\n---\n\n".join(f"[{c.heading}]\n{c.text}" for c in kept)
    prompt = f"Documentation extracts:\n\n{context}\n\n"
    if history:
        prompt += (
            f"Conversation so far (for context and phrasing only -- it is not a "
            f"source of facts):\n\n{history}\n\n"
        )
    prompt += f"Question: {question}"
    if as_list is True:
        prompt += f"\n\n{LIST_DIRECTIVE}"
    elif as_list is False:
        prompt += f"\n\n{PROSE_DIRECTIVE}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM if not history else THREAD_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        # Room for a six-point list with its "- " markers, still short of an
        # essay -- prose answers stay at four sentences regardless.
        "max_tokens": 500,
    }
    # The drafting prompt (extracts + history) is the longest call in the
    # pipeline, so an Ollama backend with a small num_ctx truncates it worst
    # here. See classifier.core.NUM_CTX.
    if NUM_CTX:
        payload["options"] = {"num_ctx": int(NUM_CTX)}

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            headers=auth_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except requests.RequestException as exc:
        return Answer(chunks=kept, error=f"request failed: {exc}")
    except (KeyError, IndexError, ValueError) as exc:
        return Answer(chunks=kept, error=f"malformed response: {exc}")

    # The refusal may arrive bare, wrapped in a sentence, dressed up as an
    # answer that reports the documents as silent, or trailing after a partial
    # answer -- see `ground`.
    text = ground(text)
    if not text:
        return Answer(text="", chunks=kept, grounded=False)

    # The instruction above is a request, not a guarantee -- a small model
    # often still answers a multi-part question in one paragraph. Split it.
    if as_list:
        text = as_bullets(text)

    return Answer(text=text, chunks=kept, grounded=True)


def default_docs() -> list[pathlib.Path]:
    """Every supported document under data/docs and data/raw."""
    found: list[pathlib.Path] = []
    for directory in (DOCS_DIR, RAW_DIR):
        if not directory.is_dir():
            continue
        found += [
            p
            for p in directory.rglob("*")
            # ~$foo.docx is Word's lock file for an open document, not content.
            if p.suffix.lower() in SUFFIXES and not p.name.startswith("~$")
        ]
    return sorted(found)
