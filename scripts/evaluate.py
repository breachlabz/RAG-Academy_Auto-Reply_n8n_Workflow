"""Run the classifier over a holdout set and sweep the routing threshold.

    python scripts/evaluate.py data/holdout.jsonl

The metric that matters is precision on the RAG route: a false positive means
the pipeline auto-replied to something it should not have. Recall tells you
what that precision costs in human queue volume -- pick a threshold where the
queue is actually staffable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from classifier.core import LABELS, LARGE_MODEL, classify, route  # noqa: E402

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


def normalize_gold(gold: dict) -> dict:
    """Gold labels in the current two-label schema. Holdout files generated
    before `administrative` and `spam` were merged into `non_academic` still
    carry the old keys; either of them set means non_academic."""
    if "non_academic" in gold:
        return gold
    return {
        "academic": gold["academic"],
        "non_academic": bool(gold.get("administrative") or gold.get("spam")),
    }


def is_rag_eligible(gold: dict) -> bool:
    """Gold-truth answer to "should this have been auto-replied?"."""
    return gold["academic"] and not gold["non_academic"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("holdout", type=pathlib.Path)
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--save", type=pathlib.Path, help="write per-email results as JSONL")
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="print the decision for every email as it is classified",
    )
    ap.add_argument(
        "--only", help="restrict to one spec, e.g. --only mixed",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.holdout.read_text().splitlines() if line.strip()]
    for row in rows:
        row["gold"] = normalize_gold(row["gold"])
    if args.only:
        rows = [r for r in rows if r["spec"] == args.only]
        if not rows:
            print(f"no emails with spec={args.only}", file=sys.stderr)
            return 1
    kwargs = {"model": LARGE_MODEL} if args.large else {}

    if args.verbose:
        print(f"{'#':>3} {'':1} {'type':<20} {'decision':<8} {'acad':>5} "
              f"{'non-ac':>6}  email")

    scored, errors = [], 0
    for i, row in enumerate(rows, 1):
        result = classify(row["body"], **kwargs)
        if not result.ok:
            errors += 1
        scored.append((row, result))

        if args.verbose:
            decision = route(result)
            expected = "rag" if is_rag_eligible(row["gold"]) else "human"
            mark = "." if decision == expected else "X"
            probs = result.probs
            cells = (
                f"{probs['academic']:>5.2f} {probs['non_academic']:>6.2f}"
                if result.ok else f"{'--':>5} {'--':>6}"
            )
            body = " ".join(row["body"].split())[:60]
            print(f"{i:>3} {mark} {row['spec']:<20} {decision:<8} {cells}  {body}")
        else:
            print(f"\r  {i}/{len(rows)}", end="", file=sys.stderr)

    if not args.verbose:
        print(file=sys.stderr)

    if args.save:
        with args.save.open("w") as fh:
            for row, result in scored:
                fh.write(json.dumps({
                    "body": row["body"], "spec": row["spec"],
                    "flavor": row.get("flavor"), "gold": row["gold"],
                    "flags": result.flags, "probs": result.probs, "error": result.error,
                }) + "\n")

    eligible = sum(is_rag_eligible(r["gold"]) for r, _ in scored)
    print(f"\n{len(scored)} emails, {eligible} RAG-eligible, {errors} errors\n")

    print(f"{'thresh':>7}  {'prec':>6}  {'recall':>6}  {'→rag':>5}  {'leaks':>5}  {'→human':>7}")
    print("  " + "-" * 45)
    for t in THRESHOLDS:
        to_rag = [(r, res) for r, res in scored if route(res, threshold=t) == "rag"]
        correct = [1 for r, _ in to_rag if is_rag_eligible(r["gold"])]
        prec = len(correct) / len(to_rag) if to_rag else 1.0
        recall = len(correct) / eligible if eligible else 0.0
        leaks = len(to_rag) - len(correct)
        print(
            f"{t:>7.2f}  {prec:>6.3f}  {recall:>6.3f}  {len(to_rag):>5}  "
            f"{leaks:>5}  {len(scored) - len(to_rag):>7}"
        )

    print("\nper-label accuracy (flags only, threshold-independent):")
    for label in LABELS:
        hits = sum(1 for r, res in scored if res.ok and res.flags.get(label) == r["gold"][label])
        total = sum(1 for _, res in scored if res.ok)
        print(f"  {label:<16} {hits}/{total}  {hits / total:.3f}" if total else f"  {label}: n/a")

    print("\nleaked emails by spec (at threshold 0.9):")
    leaked: dict[str, int] = {}
    for r, res in scored:
        if route(res, threshold=0.9) == "rag" and not is_rag_eligible(r["gold"]):
            leaked[r["spec"]] = leaked.get(r["spec"], 0) + 1
    for spec, n in sorted(leaked.items(), key=lambda kv: -kv[1]) or [("none", 0)]:
        print(f"  {spec}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
