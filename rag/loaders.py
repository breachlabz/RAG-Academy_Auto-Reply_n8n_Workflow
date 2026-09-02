"""Source documents -> the heading-marked markdown the chunker expects.

`rag.core.chunk_markdown` splits on `##` boundaries and prepends the heading
trail to every stored chunk, so a loader's real job is not "get the text out",
it is "recover the headings". Text with no headings collapses into one giant
chunk that matches everything weakly and nothing well.

Each loader is deliberately dumb and dependency-light. The .docx path already
existed and lives in `docx_text.py`; this module adds .pdf and .html and routes
between them.

Extraction loss is reported, never silenced -- see `describe()`. A PDF whose
text layer is empty is a scan, and a scan contributes nothing to retrieval no
matter how many pages it has. That has to be visible at ingest time rather than
show up months later as the model refusing questions the corpus supposedly
covers.
"""

from __future__ import annotations

import pathlib
import re

# .docx is handled by docx_text (imported lazily -- the API container queries
# but never ingests, so it does not need python-docx installed to boot).
SUFFIXES = (".md", ".txt", ".docx", ".pdf", ".html", ".htm")


def _clean(text: str) -> str:
    """Collapse whitespace runs without destroying paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- PDF --------------------------------------------------------------------
#
# PDFs carry no heading styles, only font sizes, and pypdf does not expose them
# through extract_text(). Rather than guess headings from capitalisation -- which
# turns every acronym-heavy line into an h2 -- each page becomes its own section.
# That gives retrieval a real boundary and a heading trail that at least says
# which page an answer came from.
#
# A page break is not a topic break, so this is weaker than the .docx path. For
# a PDF you care about, converting it to markdown by hand still beats it.
PDF_MIN_CHARS_PER_PAGE = 20


def pdf_to_markdown(path: pathlib.Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    lines = [f"# {path.stem}", ""]

    for number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if len(text) < PDF_MIN_CHARS_PER_PAGE:
            # Image-only page. Skipped rather than emitted as an empty section:
            # a heading with no body is a chunk that can be retrieved and
            # answers nothing.
            continue
        lines += [f"## {path.stem} — page {number}", "", text, ""]

    return "\n".join(lines).strip() + "\n"


# --- HTML -------------------------------------------------------------------
#
# HTML is the one format that actually records its headings, so h1-h6 map
# straight onto the markdown the chunker wants. bs4 + lxml are already in the
# image for nothing else; this is the payoff.
_DROP_TAGS = ("script", "style", "nav", "footer", "header", "form", "noscript")


def html_to_markdown(path: pathlib.Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")

    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    root = soup.body or soup
    lines: list[str] = []

    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre"]
    ):
        text = _clean(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # h4-h6 flatten to h3: chunk_markdown only breaks at h2/h3, and a
            # deeper level would silently stop creating a boundary.
            level = min(int(element.name[1]), 3)
            lines += ["", f"{'#' * level} {text}", ""]
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines += [text, ""]

    if not any(line.startswith("#") for line in lines):
        # No headings at all -- give it one so the chunk carries the file name
        # into its embedding instead of an empty trail.
        lines = [f"# {path.stem}", ""] + lines

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


# --- Dispatch ---------------------------------------------------------------


def load_text(path: pathlib.Path) -> str:
    """Read any supported document as markdown."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        from .docx_text import to_markdown

        return to_markdown(path)
    if suffix == ".pdf":
        return pdf_to_markdown(path)
    if suffix in (".html", ".htm"):
        return html_to_markdown(path)
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"unsupported file type: {path.suffix} ({path.name})")


def describe(path: pathlib.Path) -> dict[str, object]:
    """Extraction stats for one file, for `python -m rag check`.

    The number that matters is `chars`. A 4MB PDF reporting 300 characters is a
    scan, and no retrieval parameter will rescue it.
    """
    try:
        text = load_text(path)
    except Exception as exc:  # noqa: BLE001 -- reporting, not handling
        return {"file": path.name, "error": str(exc), "chars": 0, "headings": 0}

    return {
        "file": path.name,
        "chars": len(text),
        "headings": sum(1 for line in text.splitlines() if line.startswith("#")),
        "bytes": path.stat().st_size,
    }
