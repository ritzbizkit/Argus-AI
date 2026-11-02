"""Extract text from the source PDFs in `data/raw` into `data/processed`.

Run before `clean_text.py`. Existing outputs are skipped unless --force is
given, so re-running after adding one PDF does not redo the large ones.

Usage:
    python extract_text.py [--force] [--raw-dir DIR] [--out-dir DIR]
"""

import argparse
import sys
from pathlib import Path

try:  # PyMuPDF >= 1.24 prefers this name; `fitz` is the legacy alias.
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract(pdf_path: Path) -> tuple[str, int]:
    """Return the document text and the number of pages yielding nothing.

    Uses PyMuPDF rather than pypdf. On the GDPR source, pypdf inserted spurious
    spaces inside words — "Ar ticle", "Ha ving regar d", "Committ ee" — which
    silently corrupted roughly a fifth of that document's tokens. PyMuPDF reads
    it correctly, and this also matches the extractor app.py already uses at
    runtime.
    """
    pages, blank = [], 0
    with pymupdf.open(pdf_path) as document:
        for page in document:
            text = page.get_text() or ""
            if not text.strip():
                blank += 1
            pages.append(text)
    return "\n".join(pages), blank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-extract existing outputs.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        print(f"No such directory: {args.raw_dir}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(args.raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.raw_dir}", file=sys.stderr)
        return 1

    extracted, skipped, failed = 0, 0, []

    for pdf in pdfs:
        out_path = args.out_dir / f"{pdf.stem}.txt"
        if out_path.exists() and not args.force:
            print(f"Skipping {pdf.name} (already extracted; --force to redo)")
            skipped += 1
            continue

        print(f"Reading {pdf.name}...", flush=True)
        try:
            text, blank = extract(pdf)
        except Exception as exc:  # a corrupt or unreadable PDF
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed.append(pdf.name)
            continue

        if not text.strip():
            print(
                f"  FAILED: no text layer — {pdf.name} may be a scanned image "
                "and needs OCR.",
                file=sys.stderr,
            )
            failed.append(pdf.name)
            continue

        out_path.write_text(text, encoding="utf-8")
        note = f", {blank} page(s) with no text" if blank else ""
        print(f"  {len(text.split()):,} words{note} -> {out_path}")
        extracted += 1

    print(f"\nExtracted {extracted}, skipped {skipped}, failed {len(failed)}.")
    if failed:
        # Exit non-zero so a failure is not mistaken for a clean run — this is
        # how the NIST document went missing from the corpus unnoticed.
        print("Failed: " + ", ".join(failed), file=sys.stderr)
        return 1

    print("Next: python clean_text.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
