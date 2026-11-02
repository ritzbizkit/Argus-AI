"""Text preparation helpers for the Argus AI compliance auditor.

These functions are deliberately free of Streamlit, torch and transformers
imports so they can be imported and tested on their own.
"""

import re

# Boilerplate that appears in the header or footer of every page of the
# source policy PDFs and carries no compliance signal.
_BOILERPLATE = re.compile(r"(OFFICE FOR CIVIL RIGHTS)+")
_PAGE_MARKER = re.compile(r"Page \d+ of \d+")
# A word broken across a line break, e.g. "com- pliance". Joining these back
# together matters because the tokeniser treats the fragments as separate words.
_SPLIT_WORD = re.compile(r"(\w+)-\s+(\w+)")
_WHITESPACE = re.compile(r"\s+")

DEFAULT_CHUNK_SIZE = 150


def clean_pdf_text(text: str) -> str:
    """Normalise text extracted from an uploaded policy PDF."""
    text = _BOILERPLATE.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalise_corpus_text(text: str) -> str:
    """Normalise a raw training-corpus document before tokenisation.

    Removes page markers, rejoins words split across line breaks, drops
    non-ASCII characters that commonly survive PDF extraction, and collapses
    runs of whitespace.
    """
    text = _PAGE_MARKER.sub("", text)
    text = _SPLIT_WORD.sub(r"\1\2", text)
    text = text.encode("ascii", "ignore").decode()
    return _WHITESPACE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split text into consecutive chunks of at most `chunk_size` words.

    Chunks are the unit the semantic auditor embeds and scores, so a chunk
    needs to be long enough to carry a complete control statement and short
    enough that one statement is not diluted by unrelated text.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
