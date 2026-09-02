"""Turn a delivered email into the text the rest of the pipeline should see.

This was JavaScript in an n8n Code node until it moved here. Two reasons it had
to move: a Code node cannot be unit-tested or reused, and the moment a second
mail source exists (Gmail, IMAP) the same forty lines get copy-pasted into a
second workflow and the two drift.

What it does, and why each part earns its place:

**HTML to plain text.** Graph hands back HTML for most mail. Both the classifier
and bge-m3 do noticeably better on prose than on markup, and tags in the
embedding input are pure noise.

**Dropping the quoted history.** This is the part that matters. A reply carries
the entire previous exchange beneath it, and that text is the *previous*
question -- feed it in and retrieval is dragged back to whatever was asked last
time, while the classifier re-reads old content as though it were new. The
thread store is what carries conversation context now, deliberately and in one
place; the quoted block is the same information arriving again, unlabelled and
unbounded.
"""

from __future__ import annotations

import html
import re

# Where a reply quotes what came before. Cut at the earliest marker present.
# Deliberately conservative: each pattern is anchored to a line start, so a
# sentence merely *containing* "wrote:" does not truncate the email.
QUOTE_MARKERS = (
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}", re.I | re.M),
    re.compile(r"^\s*From:.*\n\s*Sent:", re.I | re.M),
    re.compile(r"^\s*On .{0,80}\bwrote:\s*$", re.I | re.M),
    re.compile(r"^\s*_{10,}\s*$", re.M),
)

_BLOCK_START = re.compile(r"<(p|div|tr|li|ul|ol|h[1-6]|blockquote)\b[^>]*>", re.I)
_BLOCK_END = re.compile(r"</(p|div|tr|li|ul|ol|h[1-6]|blockquote)>", re.I)
_BREAK = re.compile(r"<br\s*/?>", re.I)
_DROP = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(markup: str) -> str:
    """Markup to prose. Block boundaries become newlines so paragraphs survive.

    Both the opening and closing tag matter, not just the closing one: a
    nested list is <li>text1<ul><li>text2</li></ul></li>, and text1's own
    </li> does not appear until *after* the nested block. An end-only rule
    puts no newline between text1 and the <ul><li> that immediately follows
    it, so "certificate?" and "If so: in what form...?" ran together as one
    sentence with no separator -- a bare, un-nested list was fine, since there
    a </li> always sits between two <li> siblings.
    """
    text = _DROP.sub("", markup)
    text = _BREAK.sub("\n", text)
    text = _BLOCK_START.sub("\n", text)
    text = _BLOCK_END.sub("\n", text)
    text = _TAG.sub("", text)
    # html.unescape handles the full entity table, including the numeric forms
    # the hand-rolled JS version only partly covered.
    return html.unescape(text)


def strip_quoted(text: str) -> str:
    """Everything above the first quotation marker, minus any '>' lines."""
    cut = len(text)
    for marker in QUOTE_MARKERS:
        found = marker.search(text)
        if found and found.start() < cut:
            cut = found.start()
    kept = text[:cut].splitlines()
    return "\n".join(line for line in kept if not line.lstrip().startswith(">"))


def tidy(text: str) -> str:
    """Trailing spaces gone, runs of blank lines collapsed to one.

    Non-breaking spaces are folded to ordinary ones first. &nbsp; is everywhere
    in mail composed by Outlook and Word, and it survives html.unescape as
    U+00A0 -- invisible on screen, a different token to the model, and a
    different string to any code that splits on " ".
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def to_plain_text(body: str, *, is_html: bool = True) -> str:
    """The whole pipeline: markup out, quoted history out, whitespace tidied."""
    return tidy(strip_quoted(html_to_text(body) if is_html else body))
