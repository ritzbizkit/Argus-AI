import pytest

from text_processing import chunk_text, clean_pdf_text, normalise_corpus_text


class TestCleanPdfText:
    def test_collapses_whitespace_and_newlines(self):
        assert clean_pdf_text("a  b\n\nc\t d") == "a b c d"

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_pdf_text("\n  policy text  \n") == "policy text"

    def test_removes_repeated_page_boilerplate(self):
        raw = "OFFICE FOR CIVIL RIGHTSOFFICE FOR CIVIL RIGHTS Access is logged."
        assert clean_pdf_text(raw) == "Access is logged."

    def test_leaves_ordinary_text_untouched(self):
        assert clean_pdf_text("MFA is mandatory.") == "MFA is mandatory."

    def test_handles_empty_input(self):
        assert clean_pdf_text("") == ""


class TestNormaliseCorpusText:
    def test_rejoins_word_split_across_line_break(self):
        # Tokenisation accuracy depends on this: "com- pliance" must not
        # survive as two tokens.
        assert normalise_corpus_text("com- pliance is required") == "compliance is required"

    def test_removes_page_markers(self):
        assert normalise_corpus_text("Page 3 of 12 Encryption at rest.") == "Encryption at rest."

    def test_drops_non_ascii_characters(self):
        assert normalise_corpus_text("café — secure") == "caf secure"

    def test_collapses_whitespace(self):
        assert normalise_corpus_text("a\n\n\nb") == "a b"

    def test_does_not_join_across_a_genuine_hyphenated_pair(self):
        # No whitespace after the hyphen, so the pair is left alone.
        assert normalise_corpus_text("role-based access") == "role-based access"


class TestChunkText:
    def test_splits_on_the_requested_word_count(self):
        assert chunk_text("a b c d e", chunk_size=2) == ["a b", "c d", "e"]

    def test_returns_a_single_chunk_when_text_is_shorter_than_the_size(self):
        assert chunk_text("a b", chunk_size=10) == ["a b"]

    def test_preserves_every_word(self):
        words = [f"w{i}" for i in range(97)]
        chunks = chunk_text(" ".join(words), chunk_size=10)
        assert " ".join(chunks).split() == words

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []

    def test_rejects_a_chunk_size_below_one(self):
        with pytest.raises(ValueError):
            chunk_text("a b c", chunk_size=0)
