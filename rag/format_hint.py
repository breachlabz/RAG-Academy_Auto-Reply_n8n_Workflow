"""Decide whether a reply should be a bulleted list or plain prose.

SYSTEM tells the model to use a list "when the enquirer wrote their query in
points" -- but a small model follows that unreliably and defaults to prose. So
the decision is made here, from the enquirer's own text, and handed to
`answer()` as an explicit per-request instruction rather than left to the model
to infer. `answer()` also runs `as_bullets()` on the result as a net for when
the model ignores the instruction anyway.

The decision is deliberately conservative: two clauses in one sentence usually
read better as prose, so it takes a real structural signal -- a list the
enquirer wrote, an explicit "in points" ask, a set-shaped question, or three or
more separate questions -- to switch to a list.
"""

from __future__ import annotations

import re

# A numbered list in the enquirer's message: "1. ...\n2. ..." / "1) ...".
_NUMBERED = re.compile(r"(?m)^\s*\(?[1-9][.)]\s+\S.*(?:\n|$)\s*\(?[2-9][.)]\s+\S")
# A bulleted list they wrote themselves.
_BULLETED = re.compile(r"(?m)^\s*[-*•]\s+\S")
# An explicit ask for list / pointwise format.
_ASKS_LIST = re.compile(
    r"\b(?:list|bullet(?:ed|s|\s*point[s]?)?|point(?:s|wise|\s*wise|\s*by\s*point|"
    r"\s*form|\s*format)|in\s+points|itemi[sz]e|enumerate|"
    r"break\s+(?:it|this|them|down)|one\s+by\s+one|point-wise)\b",
    re.I,
)
# A question whose natural answer is a set of items.
_SET_QUESTION = re.compile(
    r"\bwhat\s+are\b|\bwhich\s+(?:ones?|modules?|levels?|courses?|documents?|"
    r"kits?|dates?)\b|\b(?:all|each|every|both|list)\s+(?:of\s+)?(?:the\s+)?"
    r"(?:levels?|modules?|prerequisites?|requirements?|steps?|options?|dates?|"
    r"documents?|kits?|topics?|sections?)\b",
    re.I,
)
# Interrogative pivots, for spotting a compound one-sentence question.
_INTERROG = re.compile(
    r"\b(?:what|how|when|where|which|who|why|do|does|did|is|are|was|were|can|"
    r"could|will|would|should)\b",
    re.I,
)


def wants_list(text: str) -> bool:
    """True when the enquirer's message asks for, or is shaped as, several items."""
    if not text:
        return False
    if _NUMBERED.search(text) or _BULLETED.search(text):
        return True
    if _ASKS_LIST.search(text):
        return True
    if _SET_QUESTION.search(text):
        return True
    if text.count("?") >= 2:
        return True
    # One compound question: "what does X cover, and how long, and the cost?"
    return (
        text.count("?") >= 1
        and text.lower().count(" and ") >= 2
        and len(_INTERROG.findall(text)) >= 3
    )


def resolve(mode: str, text: str) -> bool | None:
    """Map a caller's format mode to the `as_list` argument of `answer()`.

    "list" -> True, "prose" -> False, anything else ("auto", "", None) -> decide
    from `text`, which yields True only on a clear signal and otherwise None so
    the model's own SYSTEM-rule judgement still applies.
    """
    mode = (mode or "auto").strip().lower()
    if mode == "list":
        return True
    if mode == "prose":
        return False
    return True if wants_list(text) else None


LIST_DIRECTIVE = (
    'Format: the enquirer asked for several things or for a list. Reply as a '
    'bulleted list -- one "- " item per point, at most six points, no heading '
    'or preamble before the first item.'
)
PROSE_DIRECTIVE = "Format: reply in plain prose, at most four sentences, with no list."

# A sentence boundary: terminal punctuation, then whitespace, then a capital or
# an opening bracket. "Level 2.3 covers X" does not match -- the '.' is followed
# by a digit, not whitespace -- so version numbers and "e.g." survive.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
# Already a list: a line starting with a bullet or a number marker.
_IS_LIST = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")


def as_bullets(text: str, *, max_points: int = 6, min_len: int = 15) -> str:
    """Coerce a prose answer into a "- " list. No-op if it cannot be done safely.

    Returns `text` unchanged when it is already a list, when it is a single
    sentence (a one-item list is just a sentence), or when splitting would
    produce fragments too short to be real points. Only called by `answer()`
    when the format decision was list.
    """
    body = text.strip()
    if not body or _IS_LIST.search(body):
        return body

    # A short lead-in line the model put before an inline list ("You need the
    # following: X. Y. Z.") is kept as the lead-in; the rest becomes the list.
    lead = ""
    if ":" in body.split("\n", 1)[0]:
        head, _, rest = body.partition(":")
        if 0 < len(head) <= 60 and rest.strip():
            lead, body = head.strip() + ":", rest.strip()

    parts = [p.strip() for p in _SENTENCE.split(body) if p.strip()]
    parts = [p for p in parts if len(p) >= min_len]
    if len(parts) < 2:
        return text.strip()

    bullets = "\n".join(f"- {p}" for p in parts[:max_points])
    return f"{lead}\n{bullets}" if lead else bullets
