"""Walk scripted multi-turn conversations through the threading path.

`rag_eval.py` measures the refusal gate on single questions. This measures the
thing single questions cannot show: whether a **follow-up** still works once it
stops being a standalone sentence.

The conversations below are built so that every turn after the first is
deliberately unanswerable on its own. "And how long does that one take?" has no
subject; "what about the second?" has no noun. Embedded literally they retrieve
noise, so what this script really exercises is `threads.context.
standalone_question` -- the rewrite step -- and it prints the rewritten query
next to the original so you can see whether the rewrite was faithful or whether
the model quietly invented a question the enquirer never asked.

    .venv/bin/python scripts/thread_eval.py
    .venv/bin/python scripts/thread_eval.py --large      # 27B instead of the 9B
    .venv/bin/python scripts/thread_eval.py --no-rewrite # the control condition
    .venv/bin/python scripts/thread_eval.py --db ./data/threads.db --keep

By default it runs against a throwaway database in a temp directory, so it
never mixes test traffic into real Outlook conversations. `--db` plus `--keep`
points it at a real store when you want to inspect what a live thread looks
like.

What to look for, in order of importance:

  1. A follow-up that retrieves the *same* section a human would have opened.
     If turn 2 of the levels conversation does not retrieve "Level 2", the
     rewrite failed and everything downstream is noise.
  2. `grounded=0` on a turn the documents plainly cover -- the rewrite produced
     something the embedder likes less than the original.
  3. A reply that answers correctly but from the *conversation* rather than the
     extracts. THREAD_SYSTEM forbids it; this is where you would catch it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from classifier.core import LARGE_MODEL  # noqa: E402
from rag.core import answer  # noqa: E402
from threads import context as tcontext  # noqa: E402
from threads import store  # noqa: E402

# Each conversation is a list of turns. Turn 0 stands alone; every later turn is
# written to be meaningless without the ones before it.
CONVERSATIONS: list[tuple[str, list[str]]] = [
    (
        "levels-followup",
        [
            "Hello, I am interested in your vehicle hacking training. "
            "What are the different levels you offer?",
            "Thanks. What does the second one cover?",
            "And who is that level intended for?",
        ],
    ),
    (
        "pronoun-chain",
        [
            "Does your EVH training include any hands-on hardware work?",
            "Which level is that in?",
        ],
    ),
    (
        "unanswerable-followup",
        [
            "What topics are covered in the EVH training?",
            "Great. How much does it cost and when is the next intake?",
        ],
    ),
]


def run_turn(
    key: str,
    text: str,
    *,
    index: int,
    model: str | None,
    rewrite: bool,
) -> None:
    block, prior = tcontext.load(key)
    store.record_inbound(key, text, subject="thread eval")

    print(f"\n  --- turn {index + 1} " + "-" * 46)
    print(f"  enquirer : {text}")

    query = tcontext.standalone_question(text, prior) if rewrite else text
    if query != text:
        print(f"  rewritten: {query}")
    elif prior and rewrite:
        print("  rewritten: (unchanged)")

    result = answer(
        text,
        history=block,
        query=query,
        **({"model": model} if model else {}),
    )

    if not result.ok:
        print(f"  ERROR    : {result.error}")
        return

    for chunk in result.chunks:
        print(f"  chunk    : [{chunk.distance:.3f}] {chunk.heading}")

    print(f"  grounded : {int(result.grounded)}")
    body = result.text if result.grounded else "(no grounded answer)"
    print(f"  answer   : {body}")

    if result.reply:
        store.record_reply(
            key, result.reply, subject=result.subject, grounded=result.grounded
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--large", action="store_true", help=f"use {LARGE_MODEL}")
    ap.add_argument(
        "--no-rewrite",
        action="store_true",
        help="skip standalone-question rewriting -- the control condition",
    )
    ap.add_argument("--db", help="thread database (default: a temp file)")
    ap.add_argument(
        "--keep", action="store_true", help="do not delete the database afterwards"
    )
    ap.add_argument(
        "--only", help="run just the conversation with this name"
    )
    args = ap.parse_args()

    if args.db:
        db = pathlib.Path(args.db)
    else:
        db = pathlib.Path(tempfile.mkdtemp(prefix="thread-eval-")) / "threads.db"

    # store's helpers take an explicit path, but answer()/context read the
    # module-level default, so point that at the test database too.
    store.DB_PATH = db

    conversations = CONVERSATIONS
    if args.only:
        conversations = [c for c in CONVERSATIONS if c[0] == args.only]
        if not conversations:
            print(f"no conversation named {args.only!r}", file=sys.stderr)
            return 1

    print(f"database : {db}")
    print(f"model    : {LARGE_MODEL if args.large else 'default (9B)'}")
    print(f"rewriting: {'off' if args.no_rewrite else 'on'}")

    for name, turns in conversations:
        print(f"\n=== {name} " + "=" * (56 - len(name)))
        key = f"eval:{name}"
        for index, text in enumerate(turns):
            run_turn(
                key,
                text,
                index=index,
                model=LARGE_MODEL if args.large else None,
                rewrite=not args.no_rewrite,
            )

    if not args.keep and not args.db:
        for suffix in ("", "-wal", "-shm"):
            pathlib.Path(str(db) + suffix).unlink(missing_ok=True)
        db.parent.rmdir()
    else:
        print(f"\nkept: {db}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
