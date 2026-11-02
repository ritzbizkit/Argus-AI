"""Normalise the extracted corpus in `data/processed/` in place.

Run after `extract_text.py` and before `train.py`.

Usage:
    python clean_text.py [directory]
"""

import sys
from pathlib import Path

from text_processing import normalise_corpus_text

DEFAULT_DIR = Path("data/processed")


def main(directory: Path = DEFAULT_DIR) -> int:
    if not directory.is_dir():
        print(f"No such directory: {directory}", file=sys.stderr)
        print("Run extract_text.py first.", file=sys.stderr)
        return 1

    files = sorted(directory.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {directory}", file=sys.stderr)
        return 1

    for path in files:
        text = path.read_text(encoding="utf-8")
        path.write_text(normalise_corpus_text(text), encoding="utf-8")
        print(f"Cleaned: {path.name}")

    print(f"\nDone. {len(files)} file(s) normalised.")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    raise SystemExit(main(target))
