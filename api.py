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
    GET  /health

/emails/prepare and /emails/finalize are /generate-reply split in two, for the
Academy Agent (Outlook) workflow: it runs retrieval and drafting as an n8n AI
Agent node with its own Chroma tool rather than inside the endpoint, so prepare
does classify + gate + query rewrite, the agent drafts, finalize grounds the
result, and /threads/reply records it once the Outlook draft exists. The gate
still lives server-side -- prepare returns proceed=false for anything not routed
to rag, so the agent is never reached for it.

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

from fastapi import FastAPI
from pydantic import BaseModel

from classifier import DEFAULT_THRESHOLD, classify, route
from mail import to_plain_text
from rag import answer as rag_answer
from rag import resolve_format
from rag.core import EMAIL_SUBJECT, format_email, ground
from rag.format_hint import as_bullets, wants_list
from threads import context as thread_context
from threads import store as thread_store

app = FastAPI(title="Email Classifier")

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
    # Reply shape: "auto" (default) decides from the email -- a bulleted list
    # when the enquirer wrote one, asked "in points", or asked several things;
    # prose otherwise. "list" / "prose" force it.
    format: str = "auto"


class PrepareRequest(BaseModel):
    """`/emails/prepare` -- the raw Outlook trigger fields, lightly renamed.

    `body` is the message body straight from Graph (HTML unless `is_html` is
    false); the endpoint strips markup and quoted history the same way
    `/generate-reply` does. `ref` is Graph's message id, carried through
    untouched so the draft step can address `/messages/{ref}/createReply`.
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


class FinalizeRequest(BaseModel):
    """`/emails/finalize` -- what the AI Agent node produced, plus routing keys.

    `output` is the agent's raw reply text (or the bare NOT_IN_DOCUMENTS token
    it was told to emit when the documentation does not cover the question).
    `conversation_id` is the key `/emails/prepare` returned; `subject` is only a
    fallback for deriving it when that is somehow empty.

    `format` is `/emails/prepare`'s `format` echoed back: "list" runs the
    grounded body through `as_bullets` as a net for when the agent answered a
    list-shaped enquiry in one paragraph anyway; anything else leaves it alone.
    """

    output: str
    subject: str = ""
    conversation_id: str | None = None
    format: str = "prose"


class RecordReplyRequest(BaseModel):
    """`/threads/reply` -- an outbound draft to record against a conversation."""

    conversation_id: str | None = None
    reply: str
    subject: str = ""
    grounded: bool = True


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

    fresh = thread_store.record_inbound(
        key,
        email_text,
        subject=req.subject,
        message_id=req.message_id,
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
    and `subject` / `conversation_id` / `ref` (carried through to finalize, the
    thread record, and the Graph createReply call).

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

    fresh = thread_store.record_inbound(
        key, email_text, subject=req.subject, message_id=req.message_id
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
    if fresh and gate:
        query = thread_context.standalone_question(email_text, prior)

    if not fresh:
        reason = "already handled (duplicate message_id)"
    elif "error" in payload:
        reason = "unclassified"
    elif not gate:
        reason = "not routed to rag"
    else:
        reason = "ok"

    payload |= {
        "proceed": fresh and gate,
        "conversation_id": key,
        "subject": req.subject,
        "email_text": email_text,
        "query": query,
        "history": block,
        "ref": req.ref,
        "format": "list" if wants_list(email_text) else "prose",
        "thread_length": len(prior),
        "duplicate": not fresh,
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
        "subject": EMAIL_SUBJECT if grounded else "",
        "answer": body,
        "reply": format_email(body, answered=True) if grounded else "",
        "agent_output": req.output,
        "reason": "ok" if grounded else "no grounded answer in the documents",
    }


@app.post("/threads/reply")
def record_thread_reply(req: RecordReplyRequest) -> dict:
    """Record an outbound reply against the most recent unanswered turn.

    The split Outlook path needs this as its own call: `/emails/finalize` shapes
    the draft but does not store it, because the workflow records only once the
    draft is actually in the mailbox. Pairs with the `record_inbound` that
    `/emails/prepare` did earlier for the same conversation.
    """
    key = thread_store.thread_key(req.conversation_id, req.subject)
    thread_store.record_reply(
        key, req.reply, subject=req.subject, grounded=req.grounded
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
            }
            for e in exchanges
        ],
    }
