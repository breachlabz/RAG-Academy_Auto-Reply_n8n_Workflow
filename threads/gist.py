"""A one-line summary of an inbound email, for the review queue's query column.

The review UI shows a human a table of AI-drafted replies to approve or edit.
The full stripped email is often several paragraphs; the queue needs the
enquirer's actual question at a glance, not a wall of text. This is that
gist -- generated once per exchange, the first time it is listed, and cached
on the row (`threads.store.set_query_gist`) so a queue with the page open all
day does not re-summarise the same rows on every poll.
"""

from __future__ import annotations

import requests

from classifier.core import BASE_URL, SMALL_MODEL, auth_headers

SYSTEM = """Summarise an enquirer's email in one short line for a reviewer \
scanning a queue of replies.

Rules:
- At most 12 words.
- State what they are asking about, not a greeting or pleasantry.
- No quotation marks, no trailing period.
- Output the line and nothing else."""

# Same fallback shape as threads.context._truncate: word-boundary cut, not a
# mid-word chop.
_FALLBACK_CHARS = 140


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _FALLBACK_CHARS:
        return text
    return text[:_FALLBACK_CHARS].rsplit(" ", 1)[0] + "…"


def summarize_query(email_text: str, *, timeout: int = 30) -> str:
    """One line for the review queue. Falls back to truncation on any failure.

    This is a display convenience, not a routing decision -- unlike
    classification, a bad or missing summary cannot misroute an email, so a
    failed call degrades to a truncated snippet rather than blocking the queue
    from showing the row at all.
    """
    text = (email_text or "").strip()
    if not text:
        return ""
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": SMALL_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "max_tokens": 40,
            },
            headers=auth_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        gist = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return _truncate(text)

    return gist.strip('"“” ') or _truncate(text)
