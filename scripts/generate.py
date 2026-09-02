"""Generate a synthetic email set with the 27B, then split it 70/30.

Generation deliberately runs on a different model than classification. Using
one model for both gives correlated blind spots -- the emails come out phrased
the way the classifier already expects, and precision looks better than it is.

    python scripts/generate.py --out data          # generate + split
    python scripts/generate.py --out data --resume # continue an interrupted run
    python scripts/generate.py --out data --split-only

Batches append to raw.jsonl as they complete, so an interrupted run keeps its
work. Numbers from this set are a smoke test, not ground truth -- get real
email into the holdout before trusting a threshold in production.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from classifier.core import BASE_URL, LARGE_MODEL, auth_headers  # noqa: E402

# Gold labels are assigned by which spec produced the email, so the generator
# never has to self-report a label it might get wrong.
SPECS = {
    "academic_only": (
        "asks only about course content, curriculum, certification, syllabus, "
        "class schedules, or exam dates. It must not mention money, invoices, "
        "enrollment status, or personal records.",
        {"academic": True, "administrative": False, "spam": False},
    ),
    "administrative_only": (
        "concerns only payment, invoicing, refunds, enrollment status, or "
        "personal records. It must not ask anything about course content or "
        "schedules.",
        {"academic": False, "administrative": True, "spam": False},
    ),
    "mixed": (
        "asks about course content or schedules AND separately raises a "
        "payment, invoice, refund, or enrollment-status issue. Bury the "
        "administrative part mid-paragraph or at the very end so it is easy "
        "to miss.",
        {"academic": True, "administrative": True, "spam": False},
    ),
    "spam": (
        "is unrelated to the institute: marketing blasts, phishing, cold "
        "vendor outreach, newsletters, or misdirected mail.",
        {"academic": False, "administrative": False, "spam": True},
    ),
}

# These break classifiers far more often than typos do.
FLAVORS = [
    "terse, under 20 words, no greeting or signature",
    "rambling, 200+ words, buries the actual question in the middle",
    "several typos and missing apostrophes, all lowercase",
    "a forwarded chain with Fwd: headers and two levels of quoted text",
    "a reply that quotes the previous message with > markers before answering",
    "formal, with a full corporate signature block and legal disclaimer",
    "written by a parent or employer on behalf of the student",
    "mostly German or Spanish with a few English technical terms",
    "sent from a phone, fragmented, with a 'Sent from my iPhone' footer",
    "angry and demanding, with ALL CAPS emphasis",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "emails": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"body": {"type": "string"}},
                "required": ["body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["emails"],
    "additionalProperties": False,
}


def generate_batch(description: str, flavor: str, count: int, timeout: int = 300) -> list[str]:
    prompt = (
        f"Write {count} different emails sent to a vocational training "
        f"institute that teaches automotive and high-voltage technical "
        f"courses.\n\nEach email {description}\n\n"
        f"Style for this batch: {flavor}.\n\n"
        f"Vary the sender, the specific course, and the phrasing across all "
        f"{count}. Output the email bodies only -- no subject lines, no "
        f"commentary."
    )
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=auth_headers(),
        timeout=timeout,
        json={
            "model": LARGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.95,
            "max_tokens": 3000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "emails", "strict": True, "schema": SCHEMA},
            },
        },
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return [e["body"].strip() for e in json.loads(content)["emails"] if e["body"].strip()]


def flavor_targets(target: int) -> dict[str, int]:
    """Spread `target` emails across every flavor, remainder to the first few.

    Cycling flavors per batch instead means a small run only ever sees the
    first flavor or two -- a 17-email spec at batch 9 got exactly 2 of the 10
    styles, and none of the formats that actually break classifiers.
    """
    base, extra = divmod(target, len(FLAVORS))
    per = {flavor: base for flavor in FLAVORS}
    for flavor in FLAVORS[:extra]:
        per[flavor] += 1
    return {flavor: n for flavor, n in per.items() if n}


def load_raw(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split(raw: list[dict], out: pathlib.Path, holdout: float, seed: int) -> None:
    rows = list(raw)
    random.Random(seed).shuffle(rows)
    cut = int(len(rows) * (1 - holdout))
    for name, chunk in (("train", rows[:cut]), ("holdout", rows[cut:])):
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in chunk:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {len(chunk)} -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data", type=pathlib.Path)
    ap.add_argument("--batch", type=int, default=10, help="emails per model call")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep existing raw.jsonl")
    ap.add_argument("--split-only", action="store_true", help="re-split raw.jsonl")
    ap.add_argument(
        "--counts",
        default="academic_only=100,administrative_only=100,mixed=120,spam=60",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "raw.jsonl"

    if args.split_only:
        raw = load_raw(raw_path)
        if not raw:
            print(f"no rows in {raw_path}", file=sys.stderr)
            return 1
        split(raw, args.out, args.holdout, args.seed)
        return 0

    counts = dict(
        (k, int(v)) for k, v in (p.split("=") for p in args.counts.split(","))
    )

    existing = load_raw(raw_path) if args.resume else []
    if args.resume and existing:
        print(f"resuming from {len(existing)} existing rows")
    elif raw_path.is_file():
        raw_path.unlink()

    have = collections.Counter((row["spec"], row["flavor"]) for row in existing)

    with raw_path.open("a") as fh:
        for spec_name, target in counts.items():
            description, gold = SPECS[spec_name]
            for flavor, want in flavor_targets(target).items():
                made = have[(spec_name, flavor)]
                failures = 0
                while made < want:
                    take = min(args.batch, want - made)
                    try:
                        bodies = generate_batch(description, flavor, take)
                    except (requests.RequestException, ValueError, KeyError) as exc:
                        failures += 1
                        print(f"  batch failed ({spec_name}): {exc}", file=sys.stderr)
                        if failures >= 3:
                            print(f"  giving up on {spec_name}/{flavor[:20]}", file=sys.stderr)
                            break
                        continue
                    failures = 0
                    for body in bodies[:take]:
                        fh.write(json.dumps({
                            "body": body, "gold": gold,
                            "spec": spec_name, "flavor": flavor,
                        }) + "\n")
                    fh.flush()
                    made += len(bodies[:take])
                    print(f"  {spec_name} [{flavor[:28]}]: {made}/{want}", file=sys.stderr)

    split(load_raw(raw_path), args.out, args.holdout, args.seed)
    print(
        "\nHand-label holdout.jsonl before trusting it. Do not have the 80B "
        "label the set you use to judge the 9B -- you inherit its errors as "
        "ground truth on exactly the edge cases that matter."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
