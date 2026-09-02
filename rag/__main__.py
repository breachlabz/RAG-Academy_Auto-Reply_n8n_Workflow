"""CLI: load documents into the vector store, or ask them a question.

    python -m rag ingest                    # everything in data/docs + data/raw
    python -m rag ingest path/to/doc.md --reset
    python -m rag ask "does level 2 cover CAN?"
    cat email.txt | python -m rag ask
    python -m rag check                     # extraction stats before ingesting
    python -m rag manifest docs.xlsx        # sheet vs disk
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from classifier.core import LARGE_MODEL

from .core import COLLECTION, MAX_DISTANCE, TOP_K, answer, default_docs, ingest


def _ingest(args: argparse.Namespace) -> int:
    paths = [pathlib.Path(p) for p in args.paths] if args.paths else default_docs()
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"no such file: {missing[0]}", file=sys.stderr)
        return 1
    if not paths:
        print("nothing to ingest", file=sys.stderr)
        return 1

    written = ingest(paths, reset=args.reset)
    print(f"{written} chunks -> collection {COLLECTION!r}")
    for path in paths:
        print(f"  {path}")
    return 0


def _ask(args: argparse.Namespace) -> int:
    question = args.question if args.question is not None else sys.stdin.read()
    if not question.strip():
        print("empty question", file=sys.stderr)
        return 1

    kwargs = {"model": LARGE_MODEL} if args.large else {}
    if args.list:
        kwargs["as_list"] = True
    elif args.prose:
        kwargs["as_list"] = False
    result = answer(
        question, k=args.top_k, max_distance=args.max_distance, **kwargs
    )

    if not result.ok:
        print(f"error={result.error}", file=sys.stderr)
        return 1

    print(f"Subject: {result.subject}\n")
    print(result.reply)

    if not result.grounded:
        closest = min((c.distance for c in result.chunks), default=float("nan"))
        print(f"\ngrounded=0  -> human  (closest chunk {closest:.3f})", file=sys.stderr)
        return 2

    for chunk in result.chunks:
        print(f"  [{chunk.distance:.3f}] {chunk.heading}", file=sys.stderr)
    return 0


def _check(args: argparse.Namespace) -> int:
    """Report what each document actually yields, before it is ingested.

    The number that matters is `chars`. A 4MB PDF reporting 300 of them is a
    scan with no text layer, and no retrieval setting will rescue it -- the
    only symptom otherwise is the model refusing questions the corpus is
    supposed to cover.
    """
    from .loaders import describe

    paths = [pathlib.Path(p) for p in args.paths] if args.paths else default_docs()
    if not paths:
        print("no documents found", file=sys.stderr)
        return 1

    worst = 0
    for path in paths:
        stats = describe(path)
        if error := stats.get("error"):
            print(f"  ERROR  {stats['file']}: {error}")
            worst = 1
            continue
        ratio = stats["chars"] / max(stats["bytes"], 1)
        flag = "  " if stats["chars"] > 200 else "! "
        if flag == "! ":
            worst = 1
        print(
            f"{flag}{stats['file']}: {stats['chars']} chars, "
            f"{stats['headings']} headings, {ratio:.4f} chars/byte"
        )
    if worst:
        print("\n! = little or no text extracted (scanned PDF? empty file?)")
    return 0


def _manifest(args: argparse.Namespace) -> int:
    from .manifest import reconcile

    path = pathlib.Path(args.path)
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    result = reconcile(path, column=args.column, sheet=args.sheet)
    print(result.report())
    return 0 if result.complete else 2


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk and embed documents")
    p_ingest.add_argument("paths", nargs="*", help="files; omit for data/docs")
    p_ingest.add_argument(
        "--reset",
        action="store_true",
        help="drop the collection first, for when sections were deleted",
    )
    p_ingest.set_defaults(func=_ingest)

    p_ask = sub.add_parser("ask", help="answer from the documents")
    p_ask.add_argument("question", nargs="?", help="omit to read stdin")
    p_ask.add_argument("--top-k", type=int, default=TOP_K)
    p_ask.add_argument("--max-distance", type=float, default=MAX_DISTANCE)
    p_ask.add_argument(
        "--large", action="store_true", help=f"use {LARGE_MODEL} instead of the 9B"
    )
    fmt = p_ask.add_mutually_exclusive_group()
    fmt.add_argument("--list", action="store_true", help="force a bulleted reply")
    fmt.add_argument("--prose", action="store_true", help="force a prose reply")
    p_ask.set_defaults(func=_ask)

    p_check = sub.add_parser("check", help="report text extracted per document")
    p_check.add_argument("paths", nargs="*", help="files; omit for data/docs+raw")
    p_check.set_defaults(func=_check)

    p_man = sub.add_parser("manifest", help="compare an .xlsx doc list to disk")
    p_man.add_argument("path", help="the .xlsx manifest")
    p_man.add_argument("--column", help="column holding document names")
    p_man.add_argument("--sheet", help="worksheet name (default: the first)")
    p_man.set_defaults(func=_manifest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
