"""Pairing of processed documents with their reference summaries.

Shared by the training dataset and the evaluation script so both agree on
exactly which pairs exist, and so unpaired files are reported rather than
silently dropped.
"""

from pathlib import Path
from typing import NamedTuple

SUMMARY_SUFFIX = "_summary"


class Corpus(NamedTuple):
    pairs: list[tuple[Path, Path]]
    documents_without_summary: list[str]
    summaries_without_document: list[str]

    def describe(self) -> str:
        lines = [f"{len(self.pairs)} document/summary pair(s)."]
        if self.summaries_without_document:
            lines.append(
                "  Summaries with no processed document (NOT in the corpus; "
                "re-run extract_text.py): "
                + ", ".join(self.summaries_without_document)
            )
        if self.documents_without_summary:
            lines.append(
                "  Documents with no summary (skipped): "
                + ", ".join(self.documents_without_summary)
            )
        return "\n".join(lines)


def load_corpus(processed_dir, summaries_dir) -> Corpus:
    processed_dir = Path(processed_dir)
    summaries_dir = Path(summaries_dir)

    documents = {p.stem: p for p in sorted(processed_dir.glob("*.txt"))}
    summaries = {
        p.stem[: -len(SUMMARY_SUFFIX)]: p
        for p in sorted(summaries_dir.glob(f"*{SUMMARY_SUFFIX}.txt"))
    }

    paired = sorted(set(documents) & set(summaries))
    return Corpus(
        pairs=[(documents[stem], summaries[stem]) for stem in paired],
        documents_without_summary=sorted(set(documents) - set(summaries)),
        summaries_without_document=sorted(set(summaries) - set(documents)),
    )
