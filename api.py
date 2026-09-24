"""HTTP wrapper around the classifier, for n8n (or anything) to call.

n8n cannot import the Python module, so it talks to this over HTTP instead.

    POST /classify        {"text": "..."}        -> {"type": "administrative", ...}
    POST /answer          {"text": "..."}        -> the above, plus a grounded reply
    POST /generate-reply  {"email_text": ..., "conversation_id": ...}
                                                 -> the above, thread-aware
    POST /emails/prepare  {"body": ..., "ref": ...}   -> the gate + agent prompt
    POST /emails/finalize {"output": "..."}           -> ground the agent's draft
    POST /threads/reply   {"conversation_id": ..., "reply": ...}  -> record it
    GET  /threads         list stored conversations
    GET  /threads/{id}    one conversation's turns (query/reply pairs)
    GET  /review          the review queue UI (open this in a browser)
    GET  /review/queue    grounded replies awaiting review, as JSON
    GET  /review/history  already-sent replies, most recent first, as JSON
    POST /review/{id}/send  {"reply": "..."}  -> send it (Graph) and record it
    GET  /health

/emails/prepare and /emails/finalize are /generate-reply split in two, for the
Academy Agent (Outlook) workflow: it runs retrieval and drafting as an n8n AI
Agent node with its own Chroma tool rather than inside the endpoint, so prepare
does classify + gate + query rewrite, the agent drafts, finalize grounds the
result, and /threads/reply records it. The gate still lives server-side --
prepare returns proceed=false for anything not routed to rag, so the agent is
never reached for it.

/threads/reply is the workflow's last node now -- there is no Outlook Draft
step. A grounded reply lands in the review queue (/review) instead: a human
reads it, edits it if needed, and Send calls Graph directly (mail/graph.py) to
actually send it, then records the sent text back onto the row.

/answer and /generate-reply are the same pipeline; the difference is memory.
/answer is stateless and stays that way -- it is what the test form calls and
what `scripts/rag_eval.py` measures against, and a measurement whose answers
depend on what was asked before it is not a measurement. /generate-reply is the
Outlook path: it strips the HTML and quoted history from the email, records it,
loads the thread, rewrites follow-up questions before retrieval, and stores the
draft it produced.

Both keep the classifier gate *inside* the endpoint rather than exposing
retrieval on its own, so no caller can reach the RAG path around the gate. An
administrative email must not get an auto-reply just because the caller hit the
answering endpoint.

Run locally:  uvicorn api:app --port 8100
In compose:   see docker-compose.yml (talks to the chat model and chroma by name)
"""

from __future__ import annotations

import base64
import binascii
import os
import pathlib

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from classifier import DEFAULT_THRESHOLD, classify, route
from mail import graph as mail_graph
from mail import to_plain_text
from rag import answer as rag_answer
from rag import answer_followup
from rag import is_near_duplicate
from rag import resolve_format
from rag.core import EMAIL_SUBJECT, format_email, ground
from rag.format_hint import as_bullets, wants_list
from threads import context as thread_context
from threads import gist as thread_gist
from threads import store as thread_store

app = FastAPI(title="Email Classifier")

# The review queue is a Next.js app (frontend/), built with `output: "export"`
# (see frontend/next.config.js) into frontend/out/ -- plain static HTML/CSS/JS,
# no Node server runs at runtime. `basePath: "/review"` in that config makes
# every asset URL Next emits already start with /review/_next/..., matching
# the mount below; api.py only ever hands out the prebuilt files.
_REVIEW_DIST = pathlib.Path(__file__).resolve().parent / "frontend" / "out"
# check_dir=False: frontend/out/ does not exist until `npm run build` has run
# (README §X / Dockerfile's frontend-build stage). Without this, importing
# api.py before that build ever happened -- a fresh checkout, a test runner --
# would crash on startup instead of just 404ing the review page.
app.mount(
    "/review/_next",
    StaticFiles(directory=_REVIEW_DIST / "_next", check_dir=False),
    name="review-assets",
)

# "Detect the type" wants one label, but the classifier returns three
# independent flags. Collapse by priority: a payment issue outranks a course
# question (it must never be auto-answered), and spam outranks everything.
PRIORITY = ("spam", "administrative", "academic")


def primary_type(flags: dict[str, bool]) -> str:
    for label in PRIORITY:
        if flags.get(label):
            return label
    return "none"  # model set no flag -- a general inquiry we can't place


class ClassifyRequest(BaseModel):
    """`/classify` and `/answer`.

    Thread context is optional and there are two ways to supply it. Pass
    `conversation_id` and the server loads the thread itself -- the simple case,
    for a caller that has not already read it. Pass `context` and that block is
    used verbatim, for a caller that has *already* loaded the thread and,
    crucially, has already recorded this email into it: reloading there would
    hand the model the email it is classifying as its own prior context.

    `context` wins when both are given.
    """

    text: str
    threshold: float | None = None
    conversation_id: str | None = None
    context: str = ""
    # Reply shape: "auto" decides from the enquirer's text (a list they wrote,
    # an "in points" ask, several questions), "list" / "prose" force it.
    format: str = "auto"


class ReplyRequest(BaseModel):
    """The Outlook path's payload.

    Only `email_text` is required. `conversation_id` should be Outlook's
    conversationId; when it is absent the store falls back to a key derived
    from the subject, which is weaker (two enquirers asking "Course dates?"
    collapse into one thread) but better than treating every email as new.

    `message_id` should be the internetMessageId. It is what makes a re-fired
    trigger idempotent -- without it, one email delivered twice produces two
    drafts.

    `email_text` may be raw HTML straight from Graph: with `is_html` true (the
    default, since Graph hands back HTML for most mail) the endpoint strips the
    markup and the quoted history before anything sees it. Pass `is_html` false
    only when the caller has already reduced the body to plain prose.
    """

    email_text: str
    is_html: bool = True
    conversation_id: str | None = None
    subject: str = ""
    message_id: str | None = None
    threshold: float | None = None
    from_email: str = ""
    # Reply shape: "auto" (default) decides from the email -- a bulleted list
    # when the enquirer wrote one, asked "in points", or asked several things;
    # prose otherwise. "list" / "prose" force it.
    format: str = "auto"


class PrepareRequest(BaseModel):
    """`/emails/prepare` -- the raw Outlook trigger fields, lightly renamed.

    `body` is the message body straight from Graph (HTML unless `is_html` is
    false); the endpoint strips markup and quoted history the same way
    `/generate-reply` does. `ref` is Graph's message id, stored on the
    exchange row (`thread_store.record_inbound`) so the review queue's Send
    can later call `/messages/{ref}/reply` -- see mail/graph.py.
    `conversation_id` should be Outlook's conversationId -- absent, the thread
    store falls back to a subject-derived key. `message_id` should be the
    internetMessageId: it is what makes a re-fired trigger idempotent.
    """

    body: str
    is_html: bool = True
    subject: str = ""
    conversation_id: str | None = None
    message_id: str | None = None
    ref: str = ""
    threshold: float | None = None
    from_email: str = ""


class FinalizeRequest(BaseModel):
    """`/emails/finalize` -- what the AI Agent node produced, plus routing keys.

    `output` is the agent's raw reply text (or the bare NOT_IN_DOCUMENTS token
    it was told to emit when the documentation does not cover the question).
    `conversation_id` is the key `/emails/prepare` returned; `subject` is only a
    fallback for deriving it when that is somehow empty.

    `format` is `/emails/prepare`'s `format` echoed back: "list" runs the
    grounded body through `as_bullets` as a net for when the agent answered a
    list-shaped enquiry in one paragraph anyway; anything else leaves it alone.

    `exchange_id` is also `/emails/prepare`'s, echoed straight through to the
    response so Record reply can pass it to /threads/reply unchanged.
    """

    output: str
    subject: str = ""
    conversation_id: str | None = None
    format: str = "prose"
    exchange_id: int | None = None


class RecordReplyRequest(BaseModel):
    """`/threads/reply` -- an outbound draft to record against a conversation."""

    conversation_id: str | None = None
    reply: str
    subject: str = ""
    grounded: bool = True
    # n8n's `$execution.resumeUrl` for the Wait node paused right after this
    # call, when the caller is the Academy Agent workflow. See review_send().
    resume_url: str | None = None
    # Threaded all the way from /emails/prepare's record_inbound, through
    # Finalize, so this lands on the exact row it was drafted for instead of
    # thread_store.record_reply's "most recent unanswered" fallback guess.
    # None from any caller that predates this (e.g. a hand-built request) --
    # record_reply falls back to the guess in that case only.
    exchange_id: int | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True}


def _classified(
    text: str, threshold: float | None = None, context: str = ""
) -> dict:
    """Shared body of /classify. `route` is the field to branch on.

    `context` is the thread so far. It only ever affects what the model is shown
    -- the label still describes `text` alone, and the gate is unchanged: an
    administrative follow-up in an academic thread still routes to a human.
    """
    result = classify(text, context=context)
    if not result.ok:
        # Fail safe: an unclassifiable email is never "academic".
        return {"type": "unknown", "route": "human", "error": result.error}

    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    return {
        "type": primary_type(result.flags),
        "route": route(result, threshold=threshold),
        "flags": result.flags,
        "probs": {k: round(v, 4) for k, v in result.probs.items()},
        # False when the backend returned no logprobs and EC_ALLOW_UNCALIBRATED
        # let the bare flags stand in -- `probs` is then 0/1, not measured.
        "calibrated": result.calibrated,
    }


def _resolve_context(req: ClassifyRequest) -> str:
    """The thread block to classify against. See ClassifyRequest."""
    if req.context:
        return req.context
    if req.conversation_id:
        block, _ = thread_context.load(
            thread_store.thread_key(req.conversation_id)
        )
        return block
    return ""


@app.post("/classify")
def classify_email(req: ClassifyRequest) -> dict:
    return _classified(req.text, req.threshold, _resolve_context(req))


@app.post("/answer")
def answer_email(req: ClassifyRequest) -> dict:
    """Classify, then answer from the documents only if the gate allows it.

    This runs the whole pipeline rather than exposing retrieval on its own, so
    a caller cannot reach the RAG path around the classifier -- an
    administrative email must not get an auto-reply just because the caller hit
    the answering endpoint. It also keeps the n8n workflow to a single HTTP
    node, which matters given how quietly that workflow fails.

    `answered` is the field to branch on. It is true only when the email was
    routed to rag *and* the retrieved documents actually supported an answer;
    `reason` says which of those failed.

    `conversation_id` and `context` are ignored here, deliberately. This is what
    `scripts/rag_eval.py` measures against, and a measurement whose answers
    depend on what was asked before it is not a measurement. Use
    /generate-reply for anything threaded.
    """
    payload = _classified(req.text, req.threshold)
    payload |= {
        "answered": False,
        "answer": "",
        "subject": "",
        "reply": "",
        "sources": [],
        "reason": "ok",
    }

    if payload["route"] != "rag":
        # No `reply`/`subject` here on purpose. This branch is payments, records
        # and spam -- "we don't have relevant information related to this query"
        # is both untrue and unhelpful for an unpaid-invoice email, which is not
        # unanswerable, just not ours to answer automatically.
        payload["reason"] = "not routed to rag"
        return payload

    result = rag_answer(req.text, as_list=resolve_format(req.format, req.text))
    # Reported on every path that retrieved anything, not just the answered
    # one: the distance on a *refused* question is what tells you whether
    # RAG_MAX_DISTANCE is cutting too early or too late.
    if result.chunks:
        payload["closest"] = round(min(c.distance for c in result.chunks), 4)

    if not result.ok:
        # `reply` stays empty: a failed request says nothing about whether the
        # documents cover the question.
        payload["reason"] = "retrieval or model error"
        payload["error"] = result.error
        return payload

    if not result.grounded:
        # Retrieved nothing close enough, or the model refused because the
        # documents do not cover the question. The enquirer gets the
        # no-information draft, and a human still gets the email.
        payload["reason"] = "no grounded answer in the documents"
        payload["subject"] = result.subject
        payload["reply"] = result.reply
        return payload

    payload |= {
        "answered": True,
        "answer": result.text,
        "subject": result.subject,
        "reply": result.reply,
        "sources": result.sources,
    }
    return payload


@app.post("/generate-reply")
def generate_reply(req: ReplyRequest) -> dict:
    """The Outlook path: classify, gate, answer with thread context, remember.

    `answered` is still the field n8n branches on to decide whether to create a
    draft. `duplicate` is the other early exit -- a re-delivered email -- and it
    is deliberately not an error, because a trigger re-firing is normal and
    should quietly produce no second draft rather than a failed execution.
    """
    # Markup and quoted history out first -- both the classifier and bge-m3 do
    # worse on HTML, and the quoted block is the *previous* question arriving
    # again, which drags retrieval backwards. See mail/text.py.
    email_text = to_plain_text(req.email_text, is_html=req.is_html)

    key = thread_store.thread_key(req.conversation_id, req.subject)

    # Loaded *before* the new email is recorded, so `prior` is genuinely the
    # conversation up to this point and the question being asked is not also
    # sitting in its own context.
    block, prior = thread_context.load(key)

    # See /emails/prepare's identical check for why this runs before the new
    # email is recorded and why it is an embedding comparison, not a text one.
    pending = next(
        (e for e in reversed(prior) if not e.answered and not e.is_followup), None
    )

    fresh = thread_store.record_inbound(
        key,
        email_text,
        subject=req.subject,
        message_id=req.message_id,
        sender_email=req.from_email,
    )

    near_duplicate = bool(
        fresh and pending and is_near_duplicate(email_text, pending.query)
    )

    # Classified against `block`, the thread up to but not including this email.
    # Without it a follow-up is judged on its own words, and "and what does the
    # second one cover?" measures academic 0.22 / spam 0.78 -- so the gate sent
    # every second message of every conversation to a human.
    payload = _classified(email_text, req.threshold, block)
    payload |= {
        "conversation_id": key,
        "thread_length": len(prior),
        "duplicate": not fresh,
        "near_duplicate": near_duplicate,
        "answered": False,
        "answer": "",
        "subject": "",
        "reply": "",
        "sources": [],
        "reason": "ok",
    }

    if not fresh:
        payload["reason"] = "already handled (duplicate message_id)"
        return payload

    if near_duplicate:
        # Same reasoning as the duplicate-message_id branch above: the first
        # ask is still sitting unanswered on this thread, so a second draft
        # would be redundant work at best and two replies to one question at
        # worst. No draft, no error -- the existing pending turn still covers
        # this.
        payload["reason"] = "near-duplicate of a still-unanswered question in this thread"
        return payload

    if payload["route"] != "rag":
        # Payments, records and spam. No draft: "we don't have relevant
        # information" is untrue and unhelpful for an unpaid-invoice email,
        # which is not unanswerable, just not ours to answer automatically.
        payload["reason"] = "not routed to rag"
        return payload

    # A follow-up is rewritten before it is embedded; a first email is not.
    query = thread_context.standalone_question(email_text, prior)
    if query != email_text:
        payload["retrieval_query"] = query

    result = rag_answer(
        email_text,
        history=block,
        query=query,
        as_list=resolve_format(req.format, email_text),
    )

    if result.chunks:
        payload["closest"] = round(min(c.distance for c in result.chunks), 4)

    if not result.ok:
        # No reply recorded: a failed request says nothing about whether the
        # documents cover the question, so there is no draft to remember.
        payload["reason"] = "retrieval or model error"
        payload["error"] = result.error
        return payload

    payload |= {
        "answered": result.grounded,
        "answer": result.text,
        "subject": result.subject,
        "reply": result.reply,
        "sources": result.sources,
    }
    if not result.grounded:
        payload["reason"] = "no grounded answer in the documents"

    # Recorded on both branches. The no-information draft is a real reply that
    # the enquirer will have seen, and a thread that omits it reads as though
    # their question was ignored.
    if result.reply:
        thread_store.record_reply(
            key,
            result.reply,
            subject=result.subject,
            grounded=result.grounded,
        )

    return payload


@app.post("/emails/prepare")
def prepare_email(req: PrepareRequest) -> dict:
    """First half of the Outlook path, split out for the n8n AI Agent build.

    `/generate-reply` does classify -> gate -> retrieve -> draft -> remember in
    one call. The Academy Agent workflow runs the retrieval and drafting as an
    n8n AI Agent node with its own Chroma tool instead, so the HTTP side is two
    calls: this one does everything up to (not including) drafting, the agent
    drafts, and `/emails/finalize` grounds what it produced.

    Everything the workflow's later nodes read off the `Prepare` item is
    returned here: `proceed` (the gate the "Is label Academy?" node branches
    on), `history` / `email_text` / `query` / `format` (the agent's prompt),
    and `subject` / `conversation_id` / `ref` (carried through to finalize and
    the thread record; `ref` ends up on the stored exchange row for the
    review queue's Graph reply call later). `exchange_id` is the row
    `record_inbound` just opened -- Finalize and Record reply must carry it
    through unchanged so the eventual reply lands on this exact row instead
    of record_reply's "most recent unanswered" fallback guess.

    `format` is `"list"` or `"prose"`, decided from the enquirer's own wording
    the same way `/answer` decides it (`rag.format_hint.wants_list`). The agent
    node reads it to pick a bulleted or prose reply; `/emails/finalize` uses it
    as the coercion net, exactly as `rag.answer` does for the one-call path.
    """
    email_text = to_plain_text(req.body, is_html=req.is_html)
    key = thread_store.thread_key(req.conversation_id, req.subject)

    # Loaded before the new email is recorded, so `block` is the thread up to
    # but not including it -- same ordering, and same reason, as /generate-reply.
    block, prior = thread_context.load(key)

    # The still-open question on this thread, if there is one -- checked
    # against the new email below, before it is recorded, so this can never
    # find itself. A follow-up placeholder is not a real question to compare
    # against (see threads.context.render for why is_followup rows are not
    # ordinary turns).
    pending = next(
        (e for e in reversed(prior) if not e.answered and not e.is_followup), None
    )

    exchange_id = thread_store.record_inbound(
        key,
        email_text,
        subject=req.subject,
        message_id=req.message_id,
        ref=req.ref,
        sender_email=req.from_email,
    )
    fresh = exchange_id is not None

    # A second email that is really the same question again -- an impatient
    # re-ask, a reworded repeat -- while the first is still sitting
    # unanswered on this thread. Checked against meaning, not text, because
    # "when does L2 start?" and "what's the L2 start date?" share no useful
    # substring. A genuinely new question on the same thread has to pass
    # through untouched, which is why this is a real embedding comparison
    # (rag.is_near_duplicate) rather than a keyword or length heuristic.
    near_duplicate = bool(
        fresh and pending and is_near_duplicate(email_text, pending.query)
    )

    # Classified against `block` -- the thread before this email -- so a
    # follow-up is not judged on its own words alone. The gate is unchanged:
    # spam or an administrative follow-up still routes to a human.
    payload = _classified(email_text, req.threshold, block)
    gate = payload.get("route") == "rag"

    # A follow-up is rewritten into a standalone question before the agent
    # embeds it; a first email is passed through unchanged. Only worth the extra
    # model call when the email is actually going to reach the agent.
    query = email_text
    if fresh and gate and not near_duplicate:
        query = thread_context.standalone_question(email_text, prior)

    if not fresh:
        reason = "already handled (duplicate message_id)"
    elif near_duplicate:
        reason = "near-duplicate of a still-unanswered question in this thread"
    elif "error" in payload:
        reason = "unclassified"
    elif not gate:
        reason = "not routed to rag"
    else:
        reason = "ok"

    payload |= {
        "proceed": fresh and gate and not near_duplicate,
        "conversation_id": key,
        "exchange_id": exchange_id,
        "subject": req.subject,
        "email_text": email_text,
        "query": query,
        "history": block,
        "ref": req.ref,
        "format": "list" if wants_list(email_text) else "prose",
        "thread_length": len(prior),
        "duplicate": not fresh,
        "near_duplicate": near_duplicate,
        "reason": reason,
    }
    return payload


@app.post("/emails/finalize")
def finalize_email(req: FinalizeRequest) -> dict:
    """Second half of the split Outlook path: ground the agent's draft.

    The AI Agent node is told to reply with the bare token NOT_IN_DOCUMENTS
    when the documentation does not cover the question. That instruction leaks
    -- a 9B often answers in prose that reports the documents as silent instead
    -- so the same `rag.ground` net that guards `/generate-reply` runs here: it
    strips a trailing token, and rejects a bare token or a reply that opens
    with a refusal phrased as prose.

    `grounded` is the field the "Grounded answer?" node branches on. When true,
    `reply` is the ready-to-send draft (greeting, the agent's body, sign-off);
    when false it is empty and the email is left for a human. Nothing is
    written here -- the workflow records the reply only after the Outlook draft
    exists, through /threads/reply.
    """
    body = ground(req.output)
    grounded = bool(body)

    # Coercion net: the agent is asked for a bulleted reply when the enquirer
    # wrote a list, but a 9B often returns one paragraph regardless. as_bullets
    # is a no-op when the body is already a list or cannot be split safely.
    if grounded and (req.format or "").strip().lower() == "list":
        body = as_bullets(body)

    key = thread_store.thread_key(req.conversation_id, req.subject)

    return {
        "grounded": grounded,
        "conversation_id": key,
        "exchange_id": req.exchange_id,
        "subject": EMAIL_SUBJECT if grounded else "",
        "answer": body,
        "reply": format_email(body, answered=True) if grounded else "",
        "agent_output": req.output,
        "reason": "ok" if grounded else "no grounded answer in the documents",
    }


@app.post("/threads/reply")
def record_thread_reply(req: RecordReplyRequest) -> dict:
    """Record an outbound reply against the turn it was drafted for.

    The split Outlook path needs this as its own call: `/emails/finalize` shapes
    the draft but does not store it, because the workflow records only once the
    draft is actually in the mailbox. Pairs with the `record_inbound` that
    `/emails/prepare` did earlier for the same conversation -- `exchange_id`
    is that same call's row id, carried through Finalize unchanged, so this
    always targets the exact row rather than guessing.
    """
    key = thread_store.thread_key(req.conversation_id, req.subject)
    thread_store.record_reply(
        key,
        req.reply,
        subject=req.subject,
        grounded=req.grounded,
        resume_url=req.resume_url,
        exchange_id=req.exchange_id,
    )
    return {"ok": True, "conversation_id": key}


@app.get("/threads")
def list_threads() -> dict:
    return {"conversations": thread_store.conversations()}


@app.get("/threads/{conversation_id:path}")
def get_thread(conversation_id: str) -> dict:
    """One thread's turns. Path is :path -- conversationIds contain slashes."""
    exchanges = thread_store.history(conversation_id)
    summary, through = thread_store.get_summary(conversation_id)
    return {
        "conversation_id": conversation_id,
        "summary": summary,
        "summarised_through": through,
        "exchanges": [
            {
                "id": e.id,
                "query": e.query,
                "reply": e.reply,
                "grounded": e.grounded,
                "created_at": e.created_at,
                "replied_at": e.replied_at,
                "is_followup": e.is_followup,
                "sent": e.sent,
            }
            for e in exchanges
        ],
    }


class CreateFollowupRequest(BaseModel):
    """`POST /threads/{conversation_id}/followup` -- schedule one follow-up
    stage: a message the system owes the enquirer that is not a reply to
    anything they wrote and does not expect them to write back. Independent
    of any other stage already scheduled on this thread -- to queue a
    second stage, call this again with its own `due_at`; there is no chain
    to configure and no dependency between stages.
    """

    topic: str
    due_at: str
    subject: str = ""


@app.post("/threads/{conversation_id:path}/followup")
def schedule_followup(conversation_id: str, req: CreateFollowupRequest) -> dict:
    exchange_id = thread_store.create_followup(
        conversation_id, req.topic, req.due_at, subject=req.subject
    )
    return {"ok": True, "id": exchange_id, "conversation_id": conversation_id}


@app.post("/followups/process")
def process_followups() -> dict:
    """Draft every scheduled follow-up whose due_at has passed.

    Meant to be called on a schedule -- an n8n Cron workflow, a host cron
    job, whatever fires it -- not from anywhere in the enquiry/reply path.
    Idempotent to call repeatedly or concurrently: `due_followups()` only
    ever returns rows still at `reply IS NULL`, and `draft_followup` fills
    exactly the row it was given, so a follow-up already drafted by an
    overlapping run simply will not be picked up again.

    A grounded follow-up is recorded exactly like any other reply and lands
    in /review -- nothing here ever sends anything. One that could not be
    grounded (the topic has nothing to say from the documents) is left
    undrafted rather than recorded empty or invented; it stays due for the
    next run, same as a topic that genuinely has no answer never gets one
    conjured for it elsewhere in this system.
    """
    processed = []
    for row in thread_store.due_followups():
        result = answer_followup(row["query"])
        if result.grounded:
            thread_store.draft_followup(row["id"], result.reply, grounded=True)
            processed.append(
                {"id": row["id"], "conversation_id": row["conversation_id"], "drafted": True}
            )
        else:
            processed.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "drafted": False,
                    "reason": result.error or "no grounded content for this topic",
                }
            )
    return {"processed": processed}


# --- Review queue -------------------------------------------------------
#
# The human-in-the-loop step that replaces the Outlook-draft-then-manually-
# send flow. `Record reply` (/threads/reply above) already writes a grounded
# reply onto its exchange row; this is what surfaces those rows to a person,
# and what actually sends once they approve -- through mail.graph, not
# through n8n, since the Outlook OAuth2 credential the workflow would have
# used is not set up. See mail/graph.py for why that is a *separate*
# app-only credential rather than the same one.


class SendReplyRequest(BaseModel):
    """`/review/{id}/send` -- the reviewer's final text for one exchange.

    The three attachment_* fields are all optional and all-or-nothing (one
    file, see mail.graph.Attachment). content_b64 is the file read client-side
    with FileReader -- it lives in this request only: review_send passes the
    decoded bytes straight to mail_graph.send_reply and never writes them to
    disk or the DB, only attachment_name survives, onto the row's history.
    """

    reply: str
    attachment_name: str | None = None
    attachment_content_type: str | None = None
    attachment_content_b64: str | None = None


@app.get("/review")
def review_page() -> FileResponse:
    return FileResponse(_REVIEW_DIST / "index.html")


def _group_by_conversation(rows: list[dict]) -> list[dict]:
    """Fold a flat list of exchange rows into one entry per conversation, in
    first-seen order.

    `pending_review()`/`sent_history()` return one row per enquiry/reply
    turn -- correct for storage (each turn has its own `sent` flag and its
    own n8n `resume_url`, which genuinely have to stay per-turn), but wrong
    for a reviewer to look at: two pending turns from the same thread used to
    show up as two disconnected rows with nothing tying them together.
    `conversation_subject` moves up onto the group -- every row in one group
    carries the same value, so repeating it per turn was only noise.

    `other_open_threads` and `other_sent_threads` are the Case-1b signals:
    other conversations from the same sender address that still have
    something open, and other conversations already sent to them,
    respectively. Both computed for every group regardless of which section
    it ends up in -- a pending card benefits from "already sent 3 replies
    to this person" as context just as much as a History card benefits from
    "they also have something open elsewhere". Deliberately just a count +
    list for a human to read, never something this groups together or acts
    on -- see thread_store.sender_threads/sender_sent_threads for why.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        conv_id = row["conversation_id"]
        if conv_id not in groups:
            sender_email = row.get("conversation_sender_email", "") or ""
            groups[conv_id] = {
                "conversation_id": conv_id,
                "conversation_subject": row.get("conversation_subject", ""),
                "sender_email": sender_email,
                "other_open_threads": thread_store.sender_threads(
                    sender_email, exclude_conversation_id=conv_id
                ),
                "other_sent_threads": thread_store.sender_sent_threads(
                    sender_email, exclude_conversation_id=conv_id
                ),
                "exchanges": [],
            }
            order.append(conv_id)
        groups[conv_id]["exchanges"].append(
            {
                k: v
                for k, v in row.items()
                if k not in ("conversation_subject", "conversation_sender_email")
            }
        )
    return [groups[conv_id] for conv_id in order]


@app.get("/review/queue")
def review_queue() -> dict:
    """Grounded replies waiting for a human, grouped by conversation. Query
    gists are generated (and cached) here, lazily, rather than at draft time
    -- every email would otherwise pay for a summary even when the classifier
    gate or the grounding net was always going to keep it out of this queue."""
    pending = thread_store.pending_review()
    for row in pending:
        if not row.get("query_gist"):
            row["query_gist"] = thread_gist.summarize_query(row["query"])
            thread_store.set_query_gist(row["id"], row["query_gist"])
    # Surfaced so the page can show a persistent "not really sending" banner
    # the whole time REVIEW_DRY_RUN is on, not just after someone clicks Send.
    return {"pending": _group_by_conversation(pending), "dry_run": REVIEW_DRY_RUN}


@app.get("/review/history")
def review_history() -> dict:
    """Already-sent replies, grouped by conversation, most recently sent
    thread first -- the page's History section."""
    return {"history": _group_by_conversation(thread_store.sent_history())}


# UI-only testing escape hatch: skip the real Graph call so the queue/edit/
# Send flow (and the DB write it makes) can be exercised without GRAPH_* set
# up, and without sending real mail while that's being tested. Off by
# default -- REVIEW_DRY_RUN must be explicitly set to turn it on, and every
# dry-run response says so (`dry_run: true`), so it is never mistaken for a
# real send in the UI or in a log. Remove/unset it once you're done testing.
REVIEW_DRY_RUN = os.environ.get("REVIEW_DRY_RUN", "").lower() in ("1", "true", "yes")


@app.post("/review/{exchange_id}/send")
def review_send(exchange_id: int, req: SendReplyRequest) -> dict:
    """Send the reviewer's (possibly edited) reply for real, then record it.

    The Graph call comes first, on the row still in the queue: a failed send
    leaves the row exactly where it was, so the reviewer sees the error and
    can retry, rather than the row silently vanishing without the mail having
    gone anywhere. Only a successful send marks it sent.

    Skipped entirely when REVIEW_DRY_RUN is set -- see the comment above.
    """
    row = thread_store.get_exchange(exchange_id)
    if row is None:
        raise HTTPException(404, "no such exchange")
    if not row["grounded"] or row["reply"] is None:
        raise HTTPException(409, "this exchange has no AI reply to review")
    if row["sent"]:
        raise HTTPException(409, "already sent")

    reply = req.reply.strip()
    if not reply:
        raise HTTPException(422, "reply cannot be empty")

    # Decoded here, right before use, rather than kept on `req` any longer
    # than necessary -- the bytes exist only for the one Graph call below and
    # are never written anywhere; only attachment.name reaches mark_sent.
    attachment = None
    if req.attachment_name:
        try:
            content_bytes = base64.b64decode(req.attachment_content_b64 or "", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(422, f"attachment is not valid base64: {exc}") from exc
        try:
            attachment = mail_graph.Attachment(
                name=req.attachment_name,
                content_type=req.attachment_content_type or "",
                content_bytes=content_bytes,
            )
        except mail_graph.AttachmentTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc

    if not REVIEW_DRY_RUN:
        try:
            mail_graph.send_reply(row["ref"], reply, attachment=attachment)
        except mail_graph.GraphNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except mail_graph.GraphSendError as exc:
            raise HTTPException(502, str(exc)) from exc

    thread_store.mark_sent(
        exchange_id, reply, attachment_name=req.attachment_name or None
    )

    # Resume the n8n execution that's been paused (Wait node, webhook resume)
    # since Record reply, if this exchange came from that workflow. Best
    # effort only -- the mail is already sent by this point, which is what
    # actually matters; a stale/expired/unreachable resume URL (n8n
    # restarted, the 14-day cap already fired, ...) must never turn a
    # successful send into a failed response.
    resume_url = row["resume_url"]
    if resume_url:
        try:
            requests.post(resume_url, timeout=10)
        except requests.RequestException as exc:
            print(f"review_send: resume call to n8n failed for {exchange_id}: {exc}")

    return {
        "ok": True,
        "id": exchange_id,
        "sent_via_graph": not REVIEW_DRY_RUN,
        "dry_run": REVIEW_DRY_RUN,
    }
