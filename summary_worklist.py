"""Prepare and ingest the reference summaries for the section corpus.

Reference summaries have to be written by a person. This script does the parts
around that: it builds a single worklist file with every section, an inventory
of what the section contains and an excerpt, so each summary can be written
without opening the source; and it ingests the filled-in worklist back into
`data/summaries/`.

    python summary_worklist.py --init     # create/refresh the worklist
    python summary_worklist.py            # show progress
    python summary_worklist.py --apply    # write completed entries to data/summaries

Entries left blank are skipped by --apply, so the worklist can be filled in
over several sittings.
"""

import argparse
import re
from pathlib import Path

SECTIONS_DIR = Path("data/sections")
SUMMARIES_DIR = Path("data/summaries")
WORKLIST = Path("data/summary_worklist.md")

TARGET_WORDS = (30, 45)
EXCERPT_WORDS = 90
MAX_INVENTORY = 12

ENTRY = re.compile(
    r"^## (?P<slug>\S+)\n(?P<body>.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL
)
SUMMARY_FIELD = re.compile(r"^SUMMARY:(?P<text>.*?)(?=^[A-Z ]+:|\Z)", re.MULTILINE | re.DOTALL)

NIST_CONTROL = re.compile(r"\b([A-Z]{2}-\d{1,2})\s+([A-Z][A-Z0-9 ,\-/()|]{3,70}?)\s+Control:")
GDPR_ARTICLE = re.compile(r"\bArticle (\d{1,3})\s+([A-Z][^.0-9]{3,70}?)\s+1\.")


def inventory(slug: str, text: str) -> list[str]:
    """What the section covers, so a summary can be written from the worklist."""
    if slug.startswith("nist_"):
        seen, items = set(), []
        for code, name in NIST_CONTROL.findall(text):
            if code not in seen:
                seen.add(code)
                items.append(f"{code} {name.title().strip()}")
        return items
    if slug.startswith("gdpr_"):
        return [f"Art. {n} {t.strip()}" for n, t in GDPR_ARTICLE.findall(text)]
    return []


def build_entry(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    words = text.split()
    items = inventory(path.stem, text)

    listed = items[:MAX_INVENTORY]
    more = f" (+{len(items) - len(listed)} more)" if len(items) > len(listed) else ""
    covers = ("; ".join(listed) + more) if listed else "(no structured inventory)"

    return (
        f"## {path.stem}\n"
        f"WORDS: {len(words):,}\n"
        f"COVERS: {covers}\n"
        f"EXCERPT: {' '.join(words[:EXCERPT_WORDS])}...\n"
        f"SUMMARY:\n\n"
    )


def read_entries() -> dict[str, str]:
    if not WORKLIST.exists():
        return {}
    filled = {}
    for match in ENTRY.finditer(WORKLIST.read_text(encoding="utf-8")):
        field = SUMMARY_FIELD.search(match.group("body"))
        text = " ".join(field.group("text").split()) if field else ""
        filled[match.group("slug")] = text
    return filled


def cmd_init() -> int:
    sections = sorted(SECTIONS_DIR.glob("*.txt"))
    if not sections:
        print(f"No sections in {SECTIONS_DIR}. Run segment_corpus.py first.")
        return 1

    existing = read_entries()
    # Seed from summaries that already exist on disk, so previously written
    # references are not re-typed.
    for path in SUMMARIES_DIR.glob("*_summary.txt"):
        slug = path.stem[: -len("_summary")]
        if not existing.get(slug):
            existing[slug] = " ".join(path.read_text(encoding="utf-8").split())
    lo, hi = TARGET_WORDS
    header = (
        "# Reference summary worklist\n\n"
        f"Write {lo}-{hi} words under each SUMMARY:, matching the style of the\n"
        "existing reference summaries: what the section requires or grants, in\n"
        "plain prose, no bullet points. COVERS and EXCERPT are there so you do not\n"
        "need to open the source file.\n\n"
        "Leave an entry blank to skip it. Run `python summary_worklist.py --apply`\n"
        "to write the completed ones into data/summaries/.\n\n"
    )

    body = ""
    for path in sections:
        entry = build_entry(path)
        if existing.get(path.stem):  # keep anything already written
            entry = entry.replace("SUMMARY:\n\n", f"SUMMARY: {existing[path.stem]}\n\n")
        body += entry

    WORKLIST.parent.mkdir(parents=True, exist_ok=True)
    WORKLIST.write_text(header + body, encoding="utf-8")
    kept = sum(1 for s in existing.values() if s)
    print(f"Wrote {WORKLIST} with {len(sections)} section(s); kept {kept} existing summary/-ies.")
    return 0


def cmd_status() -> int:
    entries = read_entries()
    if not entries:
        print(f"No worklist at {WORKLIST}. Run with --init.")
        return 1
    done = {k: v for k, v in entries.items() if v}
    print(f"{len(done)} of {len(entries)} summaries written.")
    lo, hi = TARGET_WORDS
    for slug, text in sorted(done.items()):
        n = len(text.split())
        flag = "" if lo <= n <= hi else f"  <-- {n} words, target {lo}-{hi}"
        print(f"  {slug:<28} {n:>3}w{flag}")
    missing = [k for k, v in entries.items() if not v]
    if missing:
        print(f"\nStill to write ({len(missing)}): " + ", ".join(sorted(missing)))
    return 0


def cmd_apply() -> int:
    entries = {k: v for k, v in read_entries().items() if v}
    if not entries:
        print("Nothing to apply — no summaries written yet.")
        return 1
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    for slug, text in sorted(entries.items()):
        (SUMMARIES_DIR / f"{slug}_summary.txt").write_text(text + "\n", encoding="utf-8")
        print(f"  wrote {slug}_summary.txt ({len(text.split())} words)")
    print(f"\nApplied {len(entries)} summary/-ies to {SUMMARIES_DIR}.")
    print("Then: python evaluate.py --processed-dir data/sections")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init", action="store_true", help="Create or refresh the worklist.")
    group.add_argument("--apply", action="store_true", help="Write completed entries to data/summaries.")
    args = parser.parse_args()

    if args.init:
        return cmd_init()
    if args.apply:
        return cmd_apply()
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main())
