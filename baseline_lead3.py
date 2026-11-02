"""Print the lead-3 extractive baseline summary for each processed document.

This only shows the baseline text. To score it against the reference summaries,
use `evaluate.py`, which reports ROUGE for the baseline and for a checkpoint on
the same documents.
"""

import sys
from pathlib import Path

from corpus import load_corpus
from evaluate import lead3

PREVIEW_CHARS = 200


def main(processed_dir="data/processed", summaries_dir="data/summaries") -> int:
    corpus = load_corpus(processed_dir, summaries_dir)
    if not corpus.pairs:
        print("No document/summary pairs found.", file=sys.stderr)
        return 1

    print("--- Argus AI: lead-3 baseline summaries ---")
    for document, _ in corpus.pairs:
        summary = lead3(document.read_text(encoding="utf-8"))
        print(f"\nFILE: {document.name}")
        print(f"BASELINE: {summary[:PREVIEW_CHARS]}...")

    print("\nRun `python evaluate.py` to score these against the references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
