"""Multi-label intent detection with calibrated confidence.

Only email that is unambiguously academic-only reaches the RAG auto-reply
pipeline. Anything touching payments or records, anything the model is unsure
about, and anything that fails to classify at all goes to a human.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass, field

import requests


def _load_dotenv() -> None:
    """Populate os.environ from .env at the project root, without overriding."""
    path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()

# Everything goes through the LiteLLM proxy rather than the llama.cpp ports
# directly. LiteLLM applies enable_thinking=false for the 27B; without it the
# model spends its entire token budget on reasoning and emits no JSON at all.
# Note the proxy is published on 4001, not 4000.
BASE_URL = os.environ.get("EC_BASE_URL", "http://localhost:4001/v1")
API_KEY = os.environ.get("EC_API_KEY", "")

SMALL_MODEL = os.environ.get("EC_SMALL_MODEL", "qwen3.5-9b")
LARGE_MODEL = os.environ.get("EC_LARGE_MODEL", "qwen3.6-27b")

# Direct llama.cpp ports, for debugging the proxy out of the path. Calling the
# 27B here needs an explicit chat_template_kwargs.enable_thinking=false.
DIRECT_SMALL_URL = "http://localhost:8182/v1"
DIRECT_LARGE_URL = "http://localhost:8081/v1"


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

DEFAULT_THRESHOLD = float(os.environ.get("EC_THRESHOLD", "0.9"))
# Separately configurable, but defaults to the same bar as academic: the
# `administrative` boolean flag on its own is effectively a bare >50% vote
# (the model's greedy decode of its own schema), which is enough to override
# a 99%+ confident `academic` on a coin flip. A genuinely administrative email
# (an invoice question, a login problem) clears this by a wide margin; this
# threshold exists for the mixed-intent email sitting right on the boundary,
# so it no longer loses an academic answer it could have had to a 53% guess.
ADMIN_THRESHOLD = float(os.environ.get("EC_ADMIN_THRESHOLD", DEFAULT_THRESHOLD))

# Calibrated confidence needs token logprobs on /chat/completions. LiteLLM (over
# llama.cpp or vLLM) returns them; Ollama's OpenAI endpoint did not until ~v0.12.
# Without them the default is to treat every email as unclassified, which sends
# the whole mailbox to the human queue. EC_ALLOW_UNCALIBRATED=1 instead lets the
# model's bare true/false decode stand: the gate still fails safe (any positive
# spam/administrative flag routes to a human), but the probability thresholds no
# longer bite, so a borderline mixed-intent email is settled by one boolean
# rather than held. Leave it off for a logprob-capable backend.
ALLOW_UNCALIBRATED = os.environ.get("EC_ALLOW_UNCALIBRATED", "").lower() in (
    "1",
    "true",
    "yes",
)

# Ollama caps context at num_ctx (default 4096) and silently truncates the
# prompt to fit -- long document extracts then arrive half-missing. The robust
# fix is server-side (OLLAMA_CONTEXT_LENGTH, or a Modelfile PARAMETER num_ctx);
# where neither is reachable, EC_NUM_CTX adds an `options.num_ctx` to each
# request. Unset by default: it is an Ollama-only field and other backends
# should not see it.
NUM_CTX = os.environ.get("EC_NUM_CTX")

LABELS = ("academic", "administrative", "spam")

# The key order and wording here must stay in sync with SCHEMA below. When the
# prompt does not describe the schema, grammar-constrained decoding forces
# tokens the model finds unlikely and the logprobs stop measuring the
# classification -- they measure the grammar overriding the model instead.
SYSTEM = """You are an email router for a vocational training institute.

Output a JSON object with exactly these keys, in this order:
  "academic": true if the email asks about course content, curriculum,
              certification, syllabus, schedules, exam dates, or the
              price/fees of a course a prospective student is
              considering.
  "administrative": true if the email concerns an existing invoice,
              payment status, refund, enrollment record, or account
              issue tied to a specific enrolled student. A general
              question about what a course costs is NOT administrative
              on its own -- only billing/refund/status action on an
              existing enrollment is.
  "spam": true if the email is unrelated to the institute.

An email may set more than one flag. Set every flag independently."""

SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "boolean"} for name in LABELS},
    "required": list(LABELS),
    "additionalProperties": False,
}

# Appended to SYSTEM when the caller supplies earlier messages.
#
# A follow-up email carries almost none of its own subject matter. "And what
# does the second one cover?" was measured at academic 0.2195 / spam 0.7835 --
# routed to a human as spam -- because on its own it is a bag of stopwords with
# no institute in sight. Every threaded conversation therefore died at the gate
# on its second message, which is the exact case threading exists to serve.
#
# The rules below are what stop the fix from becoming a hole. The thread is for
# resolving what the latest email refers to, not for inheriting its label: a
# payment question asked inside an academic thread is still administrative and
# still belongs to a human. Classifying the *thread* rather than the *email*
# would auto-answer it.
CONTEXT_SYSTEM = """

You are shown the earlier messages of the conversation before the latest email.

- Classify the LATEST EMAIL ONLY. The earlier messages are context, not subject
  matter: they are there so a follow-up ("and what about the second one?") can
  be understood as asking about whatever it follows up on.
- Do not inherit the labels of the earlier messages. A follow-up about an
  existing invoice, payment status, refund, or enrollment record is
  administrative even when every message before it was academic. A follow-up
  merely asking what a course costs stays academic."""


@dataclass
class Result:
    """A classification outcome. `probs` holds P(true) per label.

    `calibrated` is False when the backend returned no token logprobs and
    EC_ALLOW_UNCALIBRATED let the flags stand in: `probs` is then a hard 0/1
    copy of `flags`, not a measured probability, and the routing thresholds
    have effectively collapsed to bare-flag votes.
    """

    flags: dict[str, bool] = field(default_factory=dict)
    probs: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    calibrated: bool = True

    @property
    def ok(self) -> bool:
        return self.error is None


def _true_probs(choice: dict) -> list[float]:
    """P(true) for each boolean field, in schema key order.

    The grammar emits exactly one true/false token per field and the schema has
    no string fields, so position maps to label unambiguously.
    """
    tokens = (choice.get("logprobs") or {}).get("content") or []
    out: list[float] = []
    for tok in tokens:
        text = tok["token"].strip()
        if text not in ("true", "false"):
            continue
        alt = next(
            (
                math.exp(a["logprob"])
                for a in tok.get("top_logprobs", [])
                if a["token"].strip() == "true"
            ),
            None,
        )
        if alt is None:
            # `true` fell outside top_logprobs, so its mass is negligible.
            alt = math.exp(tok["logprob"]) if text == "true" else 0.0
        out.append(alt)
    return out


def classify(
    body: str,
    *,
    context: str = "",
    model: str = SMALL_MODEL,
    base_url: str = BASE_URL,
    timeout: int = 90,
) -> Result:
    """Classify one email body. Never raises -- failures come back as Result.error.

    `context` is a rendered block of the earlier messages in the thread (see
    `threads.context.load`). It changes what the model is *shown*, never what it
    is asked: the label still describes `body` alone. Pass it for a follow-up,
    whose own words rarely say what it is about, and leave it empty for a first
    email, where there is nothing to resolve and the extra text only adds a way
    to be wrong.
    """
    system = SYSTEM + CONTEXT_SYSTEM if context else SYSTEM
    user = (
        f"Earlier in this conversation:\n\n{context}\n\nLatest email:\n{body}"
        if context
        else body
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 100,
        "logprobs": True,
        "top_logprobs": 5,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "intent", "strict": True, "schema": SCHEMA},
        },
    }
    if NUM_CTX:
        payload["options"] = {"num_ctx": int(NUM_CTX)}

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=auth_headers(),
            # (connect, read): a wrong/unreachable base_url should fail in
            # seconds, not hold the caller (and the n8n node in front of it) for
            # the full read budget. The model itself can still take `timeout`.
            timeout=(5, timeout),
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        flags = json.loads(choice["message"]["content"])
    except requests.RequestException as exc:
        return Result(error=f"request failed: {exc}")
    except (KeyError, IndexError, ValueError) as exc:
        return Result(error=f"malformed response: {exc}")

    has_logprobs = bool((choice.get("logprobs") or {}).get("content"))

    if not has_logprobs and ALLOW_UNCALIBRATED:
        # No logprobs and the operator has opted into flag-only routing. Use the
        # model's own true/false decode as a hard 0/1; route() still fails safe.
        missing = [label for label in LABELS if label not in flags]
        if missing:
            return Result(flags=flags, error=f"missing labels: {missing}")
        return Result(
            flags=flags,
            probs={label: 1.0 if flags[label] else 0.0 for label in LABELS},
            calibrated=False,
        )

    probs = _true_probs(choice)
    if len(probs) != len(LABELS):
        # Positional mapping is no longer trustworthy; treat as unclassified so
        # route() falls through to the human queue. (This is also the path a
        # logprob-less backend takes when EC_ALLOW_UNCALIBRATED is not set.)
        return Result(flags=flags, error=f"expected {len(LABELS)} bools, got {len(probs)}")

    return Result(flags=flags, probs=dict(zip(LABELS, probs)))


def route(result: Result, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return "rag" or "human". Every uncertain path resolves to "human".

    `administrative` is checked against ADMIN_THRESHOLD, not the raw boolean
    flag -- see the comment on ADMIN_THRESHOLD for why a bare flag has no
    margin. `spam` keeps the boolean: it has no counterpart pulling the other
    way (nothing wants a spam email routed to rag), so there is nothing for a
    margin to protect against.
    """
    if not result.ok:
        return "human"
    if result.flags.get("spam"):
        return "human"
    if result.probs.get("administrative", 0.0) >= ADMIN_THRESHOLD:
        return "human"
    if not result.flags.get("academic"):
        return "human"
    if result.probs.get("academic", 0.0) < threshold:
        return "human"
    return "rag"
