"""Reconcile the Excel list of documents-to-collect against what is on disk.

The spec has an Excel sheet naming every document that ought to be in the
corpus. That list is not the corpus -- files get renamed, never sent, or saved
as a PDF when the row says .docx -- and the gap between the two is invisible
once ingest has run, because ingest only ever sees the files that exist.

So this module answers one question: **which rows have no file, and which files
have no row.** It does not ingest anything and it is not required by ingest;
`python -m rag ingest` works whether or not a manifest exists.

The column layout is not fixed, because the sheet was not written for this
program. `read_manifest` sniffs for the column most likely to hold document
names rather than demanding a schema, and `--column` overrides it when the
sniff picks wrong.

Matching is on the stem, case-folded, punctuation-stripped: a row reading
"EVH Training - Overview v2" matches `EVH Training - Overview v2(5).docx`,
which is the version-suffix mess a real shared drive actually produces.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

from .core import default_docs

# Header text that means "this column holds the document name". Checked
# case-insensitively as a substring, longest-signal-first.
NAME_HINTS = (
    "document",
    "file",
    "title",
    "name",
    "doc",
    "source",
)

# A row whose name cell is one of these is a section divider, not a document.
_SKIP = {"", "n/a", "na", "-", "--", "tbd", "none", "total"}


def _normalise(text: str) -> str:
    """Fold a document name to something two spellings of it agree on.

    Drops the extension, any trailing "(5)" or "v2" copy marker, punctuation and
    case, so the sheet's wording and the file on disk can differ in all the ways
    they normally do and still match.
    """
    text = pathlib.Path(str(text).strip()).stem
    text = re.sub(r"\(\d+\)\s*$", "", text)          # Word's "(5)" copy suffix
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())  # punctuation -> space
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Manifest:
    """What the sheet asked for, against what is actually on disk."""

    column: str = ""
    expected: list[str] = field(default_factory=list)
    matched: dict[str, str] = field(default_factory=dict)   # row -> filename
    missing: list[str] = field(default_factory=list)        # row, no file
    unlisted: list[str] = field(default_factory=list)       # file, no row

    @property
    def complete(self) -> bool:
        return not self.missing

    def report(self) -> str:
        lines = [
            f"manifest column: {self.column!r}",
            f"{len(self.matched)}/{len(self.expected)} listed documents present",
        ]
        for row in self.missing:
            lines.append(f"  MISSING   {row}")
        for name in self.unlisted:
            lines.append(f"  UNLISTED  {name}")
        if self.complete and not self.unlisted:
            lines.append("  everything on the list is on disk, and nothing else is")
        return "\n".join(lines)


def _pick_column(header: list[str]) -> int:
    """Index of the column most likely to hold document names."""
    cells = [str(c or "").strip().lower() for c in header]
    for hint in NAME_HINTS:
        for index, cell in enumerate(cells):
            if hint in cell:
                return index
    return 0  # No recognisable header -- assume the first column.


def read_manifest(
    path: pathlib.Path,
    *,
    column: str | None = None,
    sheet: str | None = None,
) -> tuple[str, list[str]]:
    """Return (column_name, document_names) from an .xlsx manifest."""
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    worksheet = workbook[sheet] if sheet else workbook.active

    rows = [list(r) for r in worksheet.iter_rows(values_only=True)]
    if not rows:
        return ("", [])

    header, *body = rows
    header_text = [str(c or "").strip() for c in header]

    if column:
        lowered = [c.lower() for c in header_text]
        if column.lower() not in lowered:
            raise ValueError(
                f"no column {column!r} in {path.name}; found: {header_text}"
            )
        index = lowered.index(column.lower())
    else:
        index = _pick_column(header)

    names = []
    for row in body:
        if index >= len(row):
            continue
        value = str(row[index] or "").strip()
        if value.lower() not in _SKIP:
            names.append(value)

    label = header_text[index] if index < len(header_text) else f"column {index}"
    return (label or f"column {index}", names)


def reconcile(
    manifest_path: pathlib.Path,
    *,
    column: str | None = None,
    sheet: str | None = None,
    paths: list[pathlib.Path] | None = None,
) -> Manifest:
    """Compare the sheet against data/docs + data/raw."""
    label, expected = read_manifest(manifest_path, column=column, sheet=sheet)
    on_disk = paths if paths is not None else default_docs()

    by_key: dict[str, str] = {}
    for path in on_disk:
        by_key.setdefault(_normalise(path.name), path.name)

    result = Manifest(column=label, expected=expected)
    used: set[str] = set()

    for row in expected:
        key = _normalise(row)
        filename = by_key.get(key)
        if filename is None:
            # Second pass: the sheet often carries a longer descriptive title
            # than the file, or the other way round.
            filename = next(
                (
                    name
                    for existing, name in by_key.items()
                    if key and (key in existing or existing in key)
                ),
                None,
            )
        if filename is None:
            result.missing.append(row)
        else:
            result.matched[row] = filename
            used.add(filename)

    result.unlisted = sorted(
        path.name for path in on_disk if path.name not in used
    )
    return result
