"""Measure the gate that actually holds: does the RAG path refuse when it should.

Cosine distance cannot do this job on this corpus -- the distances for
answerable and unanswerable questions overlap, so no `RAG_MAX_DISTANCE` both
admits the good ones and rejects the bad. Everything therefore rests on the
model declining to answer, which it does in two ways: the `NOT_IN_DOCUMENTS`
token, and prose that reports the documents as silent. This script measures
both, plus the regex that catches the second.

    .venv/bin/python scripts/rag_eval.py
    .venv/bin/python scripts/rag_eval.py --large    # same set on the 27B

Two failure directions, and they are not equally bad:

  leak  -- answered something the documents do not support. This is the one
           the whole pipeline exists to prevent.
  miss  -- refused something the documents do support. Costs human queue
           volume, harms nobody.

The labels below are hand-written against the EVH document. Do not have a
model label them: "where is the training delivered?" reads unanswerable and is
not, because Level 3 is described as on-site, and a model that labelled it
would have written that mistake into the ground truth.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from classifier.core import LARGE_MODEL  # noqa: E402
from rag.core import answer, refuses_in_prose  # noqa: E402

# (question, answerable from the document)
QUESTIONS: list[tuple[str, bool]] = [
    ("What is EVH?", True),
    ("What are the three levels?", True),
    ("What are the different levels and who is each one for?", True),
    ("Does level 2 include hands-on hardware work?", True),
    ("Which modules cover cryptography?", True),
    ("Does the training cover UDS?", True),
    ("Is level 3 delivered on site?", True),
    # On-site delivery is stated for Level 3, so this is answerable in part.
    ("Where is the training delivered?", True),
    ("Does level 2 use real ECUs or mock ones?", True),
    ("Can the modules be adapted to a customer?", True),
    ("How much does the training cost?", False),
    ("What is the price of the EVH training?", False),
    ("How many days does the level 3 training run for?", False),
    ("How long is level 3 in days?", False),
    ("Who teaches level 1?", False),
    ("What is the maximum number of participants?", False),
    ("What is the CVSS score for a CAN injection attack?", False),
    ("What certification do I get at the end?", False),
    ("Is there a discount for booking two levels?", False),
]

# Prose the detector must catch, and prose it must not. The second group is the
# one that matters: these are real answers that happen to contain a negation,
# and flagging them would send good replies to the human queue for no reason.
PROSE_CASES: list[tuple[str, bool]] = [
    ("The provided documentation does not specify the number of days.", True),
    ("Level 2 is not explicitly detailed in the provided text.", True),
    ("The provided documentation does not contain information about who teaches Level 1.", True),
    ("The extracts do not mention a price.", True),
    ("This information is not provided in the given material.", True),
    ("The context does not describe a certification.", True),
    ("Level 1 does not include hands-on exercises; those begin at Level 2.", False),
    ("Level 3 is tailored to client needs and does not follow a fixed syllabus.", False),
    ("The training covers CAN, UDS and reverse engineering.", False),
    ("Level 2 introduces exercises on a mock-ECU, which is not a real vehicle.", False),
    ("No prerequisites are listed for Level 1.", False),
]


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag_eval", description=__doc__)
    ap.add_argument("--large", action="store_true", help=f"use {LARGE_MODEL}")
    ap.add_argument("--verbose", action="store_true", help="print every answer")
    args = ap.parse_args()
    kwargs = {"model": LARGE_MODEL} if args.large else {}

    print("=== prose-refusal detector ===")
    prose_bad = 0
    for text, expect in PROSE_CASES:
        got = refuses_in_prose(text)
        if got != expect:
            prose_bad += 1
            print(f"  FAIL flagged={int(got)} want={int(expect)}  {text}")
    print(f"  {len(PROSE_CASES) - prose_bad}/{len(PROSE_CASES)} correct")

    print("\n=== answering ===")
    leaks, misses = [], []
    distances: list[tuple[float, bool]] = []
    for question, answerable in QUESTIONS:
        result = answer(question, **kwargs)
        if not result.ok:
            print(f"  ERROR {result.error}  {question}")
            return 1
        closest = min((c.distance for c in result.chunks), default=float("nan"))
        distances.append((closest, answerable))

        if result.grounded and not answerable:
            leaks.append(question)
            flag = "LEAK"
        elif not result.grounded and answerable:
            misses.append(question)
            flag = "miss"
        else:
            flag = "ok  "
        print(f"  {flag} d={closest:.3f} grounded={int(result.grounded)}  {question}")
        if args.verbose and result.reply:
            print(f"        {result.reply[:150]}")

    total = len(QUESTIONS)
    print(f"\n{total - len(leaks) - len(misses)}/{total} correct")
    print(f"leaks (answered without support): {len(leaks)}")
    for q in leaks:
        print(f"  - {q}")
    print(f"misses (refused with support):    {len(misses)}")
    for q in misses:
        print(f"  - {q}")

    # The overlap is the point: it is why the distance cutoff is not the gate.
    yes = [d for d, a in distances if a]
    no = [d for d, a in distances if not a]
    print(f"\nanswerable distances   : {min(yes):.3f} - {max(yes):.3f}")
    print(f"unanswerable distances : {min(no):.3f} - {max(no):.3f}")
    print(f"separable by one cutoff: {max(yes) < min(no)}")

    # Only leaks fail the run. A miss costs queue volume; a leak is the failure
    # the pipeline exists to prevent.
    return 1 if (leaks or prose_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
