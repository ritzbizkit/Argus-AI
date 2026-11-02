"""Split the large source documents into section-level sources for summarising.

Two documents dominate the corpus: NIST SP 800-53 (~198k words) and the GDPR
(~56k words). A single reference summary per document gives six training pairs
in total, which is far too few to fine-tune on. Both documents are already
organised into sections — NIST into control families, the GDPR into chapters —
so splitting on those boundaries produces many coherent, self-contained sources
without inventing anything.

This script writes the sections only. It does NOT write reference summaries:
those have to be authored, and `summary_worklist.py` prepares the worklist.

Usage:
    python segment_corpus.py [--out-dir data/sections]
"""

import argparse
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
SECTIONS_DIR = Path("data/sections")

NIST_SOURCE = "NIST.SP.800-53r5"
GDPR_SOURCE = "CELEX_32016R0679_EN_TXT"

# Control families in catalogue order, with the titles NIST gives them.
NIST_FAMILIES = [
    ("AC", "Access Control", "ACCESS CONTROL"),
    ("AT", "Awareness and Training", "AWARENESS AND TRAINING"),
    ("AU", "Audit and Accountability", "AUDIT AND ACCOUNTABILITY"),
    ("CA", "Assessment, Authorization, and Monitoring", "ASSESSMENT, AUTHORIZATION, AND MONITORING"),
    ("CM", "Configuration Management", "CONFIGURATION MANAGEMENT"),
    ("CP", "Contingency Planning", "CONTINGENCY PLANNING"),
    ("IA", "Identification and Authentication", "IDENTIFICATION AND AUTHENTICATION"),
    ("IR", "Incident Response", "INCIDENT RESPONSE"),
    ("MA", "Maintenance", "MAINTENANCE"),
    ("MP", "Media Protection", "MEDIA PROTECTION"),
    ("PE", "Physical and Environmental Protection", "PHYSICAL AND ENVIRONMENTAL PROTECTION"),
    ("PL", "Planning", "PLANNING"),
    ("PM", "Program Management", "PROGRAM MANAGEMENT"),
    ("PS", "Personnel Security", "PERSONNEL SECURITY"),
    ("PT", "PII Processing and Transparency", "PERSONALLY IDENTIFIABLE INFORMATION PROCESSING AND TRANSPARENCY"),
    ("RA", "Risk Assessment", "RISK ASSESSMENT"),
    ("SA", "System and Services Acquisition", "SYSTEM AND SERVICES ACQUISITION"),
    ("SC", "System and Communications Protection", "SYSTEM AND COMMUNICATIONS PROTECTION"),
    ("SI", "System and Information Integrity", "SYSTEM AND INFORMATION INTEGRITY"),
    ("SR", "Supply Chain Risk Management", "SUPPLY CHAIN RISK MANAGEMENT"),
]

GDPR_CHAPTERS = [
    ("I", "General provisions"),
    ("II", "Principles"),
    ("III", "Rights of the data subject"),
    ("IV", "Controller and processor"),
    ("V", "Transfers of personal data to third countries"),
    ("VI", "Independent supervisory authorities"),
    ("VII", "Cooperation and consistency"),
    ("VIII", "Remedies, liability and penalties"),
    ("IX", "Specific processing situations"),
    ("X", "Delegated acts and implementing acts"),
    ("XI", "Final provisions"),
]

# Each family opens its catalogue section with a heading of the form
# "3.17 SYSTEM AND SERVICES ACQUISITION Quick link to ... Summary Table". That
# marker is unique per family, unlike a bare control ID: "Related Controls"
# lists cite IDs from every family throughout the document.
CATALOGUE_HEADING = r"3\.\d{{1,2}}\s+{title}\s+Quick link to"

MIN_SECTION_WORDS = 100

# The last control family otherwise runs on into the back matter, which is
# reference material rather than controls and should not be summarised as if
# it were.
BACK_MATTER = re.compile(r"\b(REFERENCES|GLOSSARY|ACRONYMS|CONTROL SUMMARIES|APPENDIX)\b")


def trim_back_matter(text: str) -> str:
    match = BACK_MATTER.search(text)
    return text[: match.start()].strip() if match else text


def nist_family_boundaries(text: str) -> dict[str, int]:
    """Start offset of each control family's catalogue section."""
    boundaries: dict[str, int] = {}
    for code, _, heading in NIST_FAMILIES:
        hits = [
            m.start()
            for m in re.finditer(CATALOGUE_HEADING.format(title=re.escape(heading)), text)
        ]
        if len(hits) == 1:
            boundaries[code] = hits[0]
        elif hits:
            raise ValueError(f"{code}: expected one catalogue heading, found {len(hits)}")
    return boundaries


def split_nist(text: str) -> list[tuple[str, str, str]]:
    boundaries = nist_family_boundaries(text)
    ordered = [(c, t, boundaries[c]) for c, t, _ in NIST_FAMILIES if c in boundaries]
    ordered.sort(key=lambda row: row[2])

    sections = []
    for i, (code, title, start) in enumerate(ordered):
        end = ordered[i + 1][2] if i + 1 < len(ordered) else len(text)
        body = trim_back_matter(text[start:end])
        sections.append((f"nist_{code.lower()}", f"NIST SP 800-53 — {title} ({code})", body))
    return sections


def split_gdpr(text: str) -> list[tuple[str, str, str]]:
    positions = []
    for numeral, title in GDPR_CHAPTERS:
        match = re.search(rf"CHAPTER\s+{numeral}\b(?![IVX])", text)
        if match:
            positions.append((numeral, title, match.start()))
    positions.sort(key=lambda row: row[2])

    sections = []
    for i, (numeral, title, start) in enumerate(positions):
        end = positions[i + 1][2] if i + 1 < len(positions) else len(text)
        sections.append(
            (f"gdpr_ch{numeral.lower()}", f"GDPR — Chapter {numeral}: {title}", text[start:end].strip())
        )
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=SECTIONS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, str, str]] = []

    nist_path = args.processed_dir / f"{NIST_SOURCE}.txt"
    if nist_path.exists():
        sections += split_nist(nist_path.read_text(encoding="utf-8"))
    else:
        print(f"Missing {nist_path}; run extract_text.py first.")

    gdpr_path = args.processed_dir / f"{GDPR_SOURCE}.txt"
    if gdpr_path.exists():
        sections += split_gdpr(gdpr_path.read_text(encoding="utf-8"))
    else:
        print(f"Missing {gdpr_path}; run extract_text.py first.")

    # The remaining documents are already short enough to summarise whole.
    for path in sorted(args.processed_dir.glob("*.txt")):
        if path.stem in {NIST_SOURCE, GDPR_SOURCE}:
            continue
        sections.append((path.stem, path.stem.replace("-", " "), path.read_text(encoding="utf-8").strip()))

    written, too_short = 0, []
    for slug, title, body in sections:
        words = len(body.split())
        if words < MIN_SECTION_WORDS:
            too_short.append((slug, words))
            continue
        (args.out_dir / f"{slug}.txt").write_text(body, encoding="utf-8")
        print(f"  {slug:<28} {words:>7,} words   {title}")
        written += 1

    print(f"\nWrote {written} section(s) to {args.out_dir}.")
    if too_short:
        print("Skipped (under %d words): %s" % (MIN_SECTION_WORDS, ", ".join(f"{s} ({w})" for s, w in too_short)))
    print("Next: python summary_worklist.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
