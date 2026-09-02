"""CLI: classify an email from argv or stdin.

    python -m classifier "when is the HV module? also I havent paid"
    cat email.txt | python -m classifier
"""

from __future__ import annotations

import argparse
import sys

from .core import DEFAULT_THRESHOLD, LABELS, LARGE_MODEL, classify, route


def main() -> int:
    ap = argparse.ArgumentParser(prog="classifier", description=__doc__)
    ap.add_argument("body", nargs="?", help="email body; omit to read stdin")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument(
        "--large",
        action="store_true",
        help=f"use {LARGE_MODEL} instead of the 9B",
    )
    args = ap.parse_args()

    body = args.body if args.body is not None else sys.stdin.read()
    if not body.strip():
        ap.error("empty email body")

    kwargs = {"model": LARGE_MODEL} if args.large else {}
    result = classify(body, **kwargs)
    decision = route(result, threshold=args.threshold)

    if not result.ok:
        print(f"route={decision}  error={result.error}", file=sys.stderr)
        return 1

    probs = "  ".join(f"{name}={result.probs[name]:.2f}" for name in LABELS)
    print(f"route={decision}  {probs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
