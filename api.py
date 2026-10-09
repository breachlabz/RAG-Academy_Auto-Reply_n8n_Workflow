"""HTTP wrapper around the classifier, for n8n (or anything) to call.

n8n cannot import the Python module, so it talks to this over HTTP instead.

    POST /classify        {"text": "..."}        -> {"type": "non_academic", ...}
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
    GET  /review/manual   enquiries the documents could not answer, awaiting a hand-written reply
    POST /review/manual   {"exchange_id", "resume_url"} -> put one on that queue (the workflow)
    GET  /review/history  already-sent replies, most recent first, as JSON
    POST /review/{id}/send  {"reply": "..."}  -> hand it to n8n (Outlook draft)
    POST /review/{id}/drafted {"reply": "..."} -> the workflow's last node: draft exists, record it
    GET  /knowledge       knowledge chunks (content + JSON metadata), read-only
    POST /knowledge       {"content": ..., "title"?: ...} -> append to Additions_N.docx
    GET  /knowledge/{id}  one chunk
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
reads it, edits it if needed, and Send resumes the workflow's Wait node with
the final text; the Outlook Draft node after it creates the reply draft in the
mailbox, and the final text is recorded back onto the row.

/answer and /generate-reply are the same pipeline; the difference is memory.
/answer is stateless and stays that way -- it is what the test form calls and
what `scripts/rag_eval.py` measures against, and a measurement whose answers
depend on what was asked before it is not a measurement. /generate-reply is the
Outlook path: it strips the HTML and quoted history from the email, records it,
loads the thread, rewrites follow-up questions before retrieval, and stores the
draft it produced.

Both keep the classifier gate *inside* the endpoint rather than exposing
retrieval on its own, so no caller can reach the RAG path around the gate. A
non-academic email must not get an auto-reply just because the caller hit the
answering endpoint.

Run locally:  uvicorn api:app --port 8100
In compose:   see docker-compose.yml (talks to the chat model and chroma by name)
"""

from __future__ import annotations

import base64
import binascii
import datetime
import json
import os
import pathlib

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from classifier import DEFAULT_THRESHOLD, classify, route
from mail import to_plain_text
from rag import answer as rag_answer
from rag import answer_followup
from rag import is_near_duplicate
from rag import resolve_format
from rag import additions as rag_additions
from rag import knowledge as rag_knowledge
from rag.core import COLLECTION as RAG_COLLECTION
from rag.core import EMAIL_CLOSER_ANSWERED, EMAIL_GREETING, EMAIL_SIGNOFF
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

# "Detect the type" wants one label, but the classifier returns two
# independent flags. An email is `academic` only when it is academic and
# nothing else; everything else -- a payment issue riding along with a course
# question (which must never be auto-answered), spam, or an email the model set
# no flag on at all -- is `non_academic`.
def primary_type(flags: dict[str, bool]) -> str:
    if flags.get("academic") and not flags.get("non_academic"):
        return "academic"
    return "non_academic"


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
    response so Queue for review (UI) can pass it to /threads/reply unchanged.

    `searched` is whether the agent called its academy_docs tool at least once
    (the node sends `intermediateSteps.length > 0`). False means the model
    skipped retrieval and wrote the reply from its own head, which is never
    grounded however plausible it reads. None -- an older workflow that does
    not send the field -- leaves the decision to the text checks alone.
    """

    output: str
    subject: str = ""
    conversation_id: str | None = None
    format: str = "prose"
    exchange_id: int | None = None
    searched: bool | None = None


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
    non-academic follow-up in an academic thread still routes to a human.
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
    a caller cannot reach the RAG path around the classifier -- a
    non-academic email must not get an auto-reply just because the caller hit
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
        # No `reply`/`subject` here on purpose. This branch is non-academic
        # mail (payments, records, spam) -- "we don't have relevant information
        # related to this query" is both untrue and unhelpful for an
        # unpaid-invoice email, which is not unanswerable, just not ours to
        # answer automatically.
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
    # second one cover?" measured academic 0.22 / spam 0.78 under the old
    # three-label schema -- so the gate sent every second message of every
    # conversation to a human.
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
        # Non-academic: payments, records, spam. No draft: "we don't have relevant
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
    Outlook Draft node's reply call later). `exchange_id` is the row
    `record_inbound` just opened -- Finalize and Queue for review (UI) must carry it
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
    # a non-academic follow-up still routes to a human.
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
    when false it is empty and the email is left for a human. A draft that
    admits the documents are silent on what the enquirer asked is also false,
    even when other facts surround the admission, and so is any draft the
    agent wrote without searching the documentation (`searched: false`).
    Nothing is
    written here -- the workflow records the reply only after the Outlook draft
    exists, through /threads/reply.
    """
    # The enquiry itself, so `ground` can tell an admission about a side
    # detail (dropped, rest kept) from one about what was actually asked
    # (ungrounded -> Human queue). Read back by exchange_id rather than taken
    # from the request, so the workflow's Finalize node needs no new field.
    exchange = thread_store.get_exchange(req.exchange_id) if req.exchange_id else None
    body = ground(req.output, question=(exchange or {}).get("query") or "")
    reason = "ok" if body else "no grounded answer in the documents"

    # A reply written without a single documentation search cannot be grounded
    # in it. The agent's prompt says it MUST search first; a 9B skips the tool
    # now and then and states invented specifics in the same confident shape
    # as a real answer ("90 minutes, 40 questions" for an exam the documents
    # give as 30 and 30), which no check on the wording can tell apart.
    if body and req.searched is False:
        body = ""
        reason = "the agent answered without searching the documents"
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
        "reason": reason,
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
# send flow. `Queue for review (UI)` (/threads/reply above) already writes a grounded
# reply onto its exchange row; this is what surfaces those rows to a person,
# and what actually sends once they approve -- through mail.graph, not
# through n8n, since the Outlook OAuth2 credential the workflow would have
# used is not set up. See mail/graph.py for why that is a *separate*
# app-only credential rather than the same one.


class SendReplyRequest(BaseModel):
    """`/review/{id}/send` -- the reviewer's final text for one exchange.

    The three attachment_* fields are all optional and all-or-nothing (one
    file). content_b64 is the file read client-side with FileReader -- it
    lives in this request only: review_send passes it on to n8n for the
    Outlook draft and never writes it to disk or the DB, only attachment_name
    survives, onto the row's history.
    """

    reply: str
    attachment_name: str | None = None
    attachment_content_type: str | None = None
    attachment_content_b64: str | None = None


@app.get("/review")
def review_page() -> FileResponse:
    # no-cache: the page names its scripts by content hash, so a browser that
    # reuses an old copy of this file keeps running the previous build's code
    # after a deploy. Revalidating it on every load costs one tiny request.
    return FileResponse(
        _REVIEW_DIST / "index.html", headers={"Cache-Control": "no-cache"}
    )


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


class QueueManualRequest(BaseModel):
    """`POST /review/manual` -- the workflow's "Queue for manual reply (UI)"
    node, on the Human queue branch: the documents had no grounded answer."""

    exchange_id: int
    resume_url: str
    subject: str = ""


@app.post("/review/manual")
def review_manual_add(req: QueueManualRequest) -> dict:
    if thread_store.get_exchange(req.exchange_id) is None:
        raise HTTPException(404, "no such exchange")
    thread_store.queue_manual(req.exchange_id, req.resume_url, subject=req.subject)
    return {"ok": True, "id": req.exchange_id}


@app.get("/review/manual")
def review_manual() -> dict:
    """Enquiries with no AI reply, waiting for a person to write one. Same
    shape as /review/queue. `prefill` is the empty email shell (greeting,
    closing line, sign-off) the edit box starts from."""
    pending = thread_store.pending_manual()
    for row in pending:
        if not row.get("query_gist"):
            row["query_gist"] = thread_gist.summarize_query(row["query"])
            thread_store.set_query_gist(row["id"], row["query_gist"])
        row["manual"] = True
        row["prefill"] = format_email("", answered=True)
    return {"pending": _group_by_conversation(pending), "dry_run": REVIEW_DRY_RUN}


def _is_manual(row: dict) -> bool:
    """On the manual queue: no AI reply, and an n8n execution waiting."""
    return not row["grounded"] and row["reply"] is None and bool(row["resume_url"])


def _record_final(row: dict, reply: str, attachment_name: str | None) -> None:
    """Mark a row sent with the reviewer's final text."""
    if row["reply"] is None:
        thread_store.record_manual_reply(row["id"], reply, attachment_name=attachment_name)
    else:
        thread_store.mark_sent(row["id"], reply, attachment_name=attachment_name)


def _body_without_shell(reply: str) -> str:
    """The reply minus the greeting, closing line and sign-off, so what goes
    into the knowledge base is the answer and not the courtesy around it."""
    body = reply.strip()
    if body.startswith(EMAIL_GREETING):
        body = body[len(EMAIL_GREETING):].strip()
    if body.endswith(EMAIL_SIGNOFF):
        body = body[: -len(EMAIL_SIGNOFF)].strip()
    if body.endswith(EMAIL_CLOSER_ANSWERED):
        body = body[: -len(EMAIL_CLOSER_ANSWERED)].strip()
    return body


def _learn_from_manual(row: dict, reply: str) -> dict:
    """Add a hand-written reply to the knowledge base, exactly as
    POST /knowledge does (rag.additions), so the next enquiry
    on the same point can be answered from it. The entry is the question's
    one-line gist as its title and the reply body under it -- never the
    enquirer's own email text. Never raises: the reply is already recorded,
    and a knowledge failure must not undo or hide that."""
    body = _body_without_shell(reply)
    if not body:
        return {"error": "the reply has no body besides the greeting and sign-off"}
    gist = row.get("query_gist") or thread_gist.summarize_query(row["query"])
    try:
        added = rag_additions.add(f"Question: {gist}\n\nAnswer: {body}", gist)
    except Exception as exc:  # embedder or Chroma unreachable/rejected
        return {"error": str(exc)}
    return {"file": added["file"], "entry": added["entry"]}


@app.get("/review/history")
def review_history() -> dict:
    """Already-sent replies, grouped by conversation, most recently sent
    thread first -- the page's History section."""
    return {"history": _group_by_conversation(thread_store.sent_history())}


# UI-only testing escape hatch: skip the hand-off to n8n so the queue/edit/
# Send flow (and the DB write it makes) can be exercised without the Outlook
# credential set up, and without creating real drafts while that's tested. Off by
# default -- REVIEW_DRY_RUN must be explicitly set to turn it on, and every
# dry-run response says so (`dry_run: true`), so it is never mistaken for a
# real send in the UI or in a log. Remove/unset it once you're done testing.
# Graph refuses a file over ~3MB on the same-call attachment path the Outlook
# Draft node uses. Checked on the decoded bytes, not the base64 text.
MAX_ATTACHMENT_BYTES = 3 * 1024 * 1024

REVIEW_DRY_RUN = os.environ.get("REVIEW_DRY_RUN", "").lower() in ("1", "true", "yes")


@app.post("/review/{exchange_id}/send")
def review_send(exchange_id: int, req: SendReplyRequest) -> dict:
    """Hand the reviewer's (possibly edited) reply to n8n, then record it.

    n8n records the final text and leaves it as a reply draft in the mailbox,
    where a person presses Send. `sent` on the row means "approved on the
    review page and recorded" -- the draft is attempted but the record does
    not wait on it; `draft_error` in the response says when it failed.

    Skipped entirely when REVIEW_DRY_RUN is set -- see the comment above.
    """
    row = thread_store.get_exchange(exchange_id)
    if row is None:
        raise HTTPException(404, "no such exchange")
    if row["sent"]:
        raise HTTPException(409, "already sent")
    manual = _is_manual(row)
    if not manual and (not row["grounded"] or row["reply"] is None):
        raise HTTPException(409, "this exchange has no AI reply to review")

    reply = req.reply.strip()
    if not reply:
        raise HTTPException(422, "reply cannot be empty")

    # One optional file, passed straight through to n8n's Outlook Draft node
    # with the reply and never written here -- only its name is recorded.
    attachment = None
    if req.attachment_name:
        try:
            size = len(base64.b64decode(req.attachment_content_b64 or "", validate=True))
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(422, f"attachment is not valid base64: {exc}") from exc
        if size > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                413, f"attachment is {size} bytes, over the {MAX_ATTACHMENT_BYTES} byte limit"
            )
        attachment = {
            "name": req.attachment_name,
            "content_type": req.attachment_content_type or "application/octet-stream",
            "content_b64": req.attachment_content_b64,
        }

    # Resume the n8n execution paused at "Wait for review", handing it the
    # reviewer's final wording. Two nodes run on it: Record final draft calls
    # /review/{id}/drafted to record the text and mark this row, then Outlook
    # Draft turns it into a reply draft in the mailbox. The Wait node answers
    # only when they have finished. Recording does not depend on the draft:
    # if the row was recorded but the workflow reported an error, the reply
    # is kept as recorded and the draft failure goes back as a warning.
    draft_error = ""
    if not REVIEW_DRY_RUN:
        resume_url = row["resume_url"]
        if not resume_url:
            raise HTTPException(409, "no n8n execution is waiting on this exchange")
        try:
            resp = requests.post(
                resume_url,
                json={
                    "exchange_id": exchange_id,
                    "reply": reply,
                    "ref": row["ref"],
                    "subject": row["reply_subject"],
                    "attachment": attachment,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            draft_error = str(exc)
        if not (thread_store.get_exchange(exchange_id) or {}).get("sent"):
            raise HTTPException(
                502, f"n8n did not record the reply: {draft_error or 'no error reported'}"
            )
    else:
        _record_final(row, reply, req.attachment_name or None)

    # A reply a person had to write by hand is knowledge the documents lacked.
    knowledge = _learn_from_manual(row, reply) if manual else None

    return {
        "ok": True,
        "id": exchange_id,
        "drafted_in_outlook": not REVIEW_DRY_RUN and not draft_error,
        "draft_error": draft_error,
        "knowledge": knowledge,
        "dry_run": REVIEW_DRY_RUN,
    }


class DraftedRequest(BaseModel):
    """`/review/{id}/drafted` -- the reply as it went into the Outlook draft."""

    reply: str
    attachment_name: str | None = None


@app.post("/review/{exchange_id}/drafted")
def review_drafted(exchange_id: int, req: DraftedRequest) -> dict:
    """The workflow's Record final draft node: the reviewer's final text.

    This is what marks a row sent. It runs alongside Outlook Draft and does
    not depend on it. Idempotent -- mark_sent ignores a row that is
    already marked.
    """
    row = thread_store.get_exchange(exchange_id)
    if row is None:
        raise HTTPException(404, "no such exchange")
    reply = req.reply.strip()
    if not reply:
        raise HTTPException(422, "reply cannot be empty")
    if not row["sent"]:
        _record_final(row, reply, req.attachment_name or None)
    return {"ok": True, "id": exchange_id}


# --- Knowledge base -----------------------------------------------------
#
# A read-only view of what retrieval searches: one row per Chroma chunk in
# `knowledge_chunks` (rag/knowledge.py), with its metadata as JSON. Existing
# chunks cannot be edited or deleted here -- the .docx files are the source.
# The one write is adding: POST appends an entry to data/docs/Additions_N.docx
# (rag/additions.py, 20 entries per file) and ingests that file, so additions
# are ordinary document chunks. Same trust boundary as /review: localhost
# only, no login of its own.
#
# Chunk ids contain "#" (e.g. "EVH_Level_3_2.4.docx#3"), so callers must
# URL-encode them (%23) -- the `:path` converter keeps any "/" intact too.


class KnowledgeAddRequest(BaseModel):
    """POST /knowledge. `title` is optional; the first line of `content`
    stands in for it."""

    content: str
    title: str = ""


def _knowledge_call(fn, *args, **kwargs):
    """Map rag.knowledge's exceptions onto HTTP statuses."""
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, "no such chunk") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # embedder or Chroma unreachable/rejected
        raise HTTPException(502, f"could not update the vector store: {exc}") from exc


@app.get("/knowledge")
def knowledge_list(q: str = "", source: str = "") -> dict:
    chunks = rag_knowledge.list_chunks(q=q, source=source)
    sources = sorted(
        {str(c["metadata"].get("source", "")) for c in rag_knowledge.list_chunks()}
    )
    return {"chunks": chunks, "sources": sources, "collection": RAG_COLLECTION}


@app.post("/knowledge")
def knowledge_add(req: KnowledgeAddRequest) -> dict:
    """Append to the current data/docs/Additions_N.docx and ingest it."""
    return _knowledge_call(rag_additions.add, req.content, req.title)


# Declared before the `{chunk_id:path}` route below, which would otherwise
# swallow "export" as a chunk id.
@app.get("/knowledge/export")
def knowledge_export() -> Response:
    """Everything in the knowledge base as one downloadable JSON file.
    Read-only, same data as GET
    /knowledge, with a Content-Disposition header so a browser saves it."""
    chunks = rag_knowledge.list_chunks()
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "collection": RAG_COLLECTION,
        "exported_at": now.isoformat(timespec="seconds"),
        "count": len(chunks),
        "chunks": chunks,
    }
    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="knowledge-export-{now:%Y-%m-%d}.json"'
        },
    )


@app.get("/knowledge/{chunk_id:path}")
def knowledge_get(chunk_id: str) -> dict:
    chunk = rag_knowledge.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(404, "no such chunk")
    return chunk
