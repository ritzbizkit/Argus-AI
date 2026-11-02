from pathlib import Path

import pytest

from corpus import load_corpus


@pytest.fixture
def corpus_dirs(tmp_path):
    processed = tmp_path / "processed"
    summaries = tmp_path / "summaries"
    processed.mkdir()
    summaries.mkdir()
    return processed, summaries


def write(directory: Path, name: str, text: str = "x") -> None:
    (directory / name).write_text(text, encoding="utf-8")


class TestLoadCorpus:
    def test_pairs_documents_with_their_summaries(self, corpus_dirs):
        processed, summaries = corpus_dirs
        write(processed, "gdpr.txt")
        write(summaries, "gdpr_summary.txt")

        corpus = load_corpus(processed, summaries)

        assert [p[0].name for p in corpus.pairs] == ["gdpr.txt"]
        assert [p[1].name for p in corpus.pairs] == ["gdpr_summary.txt"]

    def test_reports_a_summary_whose_document_was_never_extracted(self, corpus_dirs):
        # This is the NIST case: the PDF and summary exist, but extraction
        # produced no text, so the pair would otherwise vanish silently.
        processed, summaries = corpus_dirs
        write(processed, "gdpr.txt")
        write(summaries, "gdpr_summary.txt")
        write(summaries, "nist_summary.txt")

        corpus = load_corpus(processed, summaries)

        assert corpus.summaries_without_document == ["nist"]
        assert len(corpus.pairs) == 1

    def test_reports_a_document_with_no_summary(self, corpus_dirs):
        processed, summaries = corpus_dirs
        write(processed, "gdpr.txt")
        write(processed, "orphan.txt")
        write(summaries, "gdpr_summary.txt")

        corpus = load_corpus(processed, summaries)

        assert corpus.documents_without_summary == ["orphan"]

    def test_pairs_are_ordered_deterministically(self, corpus_dirs):
        processed, summaries = corpus_dirs
        for stem in ["c", "a", "b"]:
            write(processed, f"{stem}.txt")
            write(summaries, f"{stem}_summary.txt")

        corpus = load_corpus(processed, summaries)

        assert [p[0].stem for p in corpus.pairs] == ["a", "b", "c"]

    def test_empty_directories_give_an_empty_corpus(self, corpus_dirs):
        processed, summaries = corpus_dirs
        corpus = load_corpus(processed, summaries)
        assert corpus.pairs == []

    def test_describe_mentions_an_unextracted_document(self, corpus_dirs):
        processed, summaries = corpus_dirs
        write(processed, "gdpr.txt")
        write(summaries, "gdpr_summary.txt")
        write(summaries, "nist_summary.txt")

        assert "nist" in load_corpus(processed, summaries).describe()
