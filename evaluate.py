"""Score summarisation quality against the reference summaries.

Reports ROUGE-1, ROUGE-2 and ROUGE-L F1 for either the lead-3 extractive
baseline or a seq2seq checkpoint, so the two can be compared on the same
documents.

Usage:
    python evaluate.py                                  # lead-3 baseline
    python evaluate.py --system model                   # ./argus_v2_bart
    python evaluate.py --system model --model-path facebook/bart-large-cnn
"""

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from corpus import load_corpus

METRICS = ["rouge1", "rouge2", "rougeL"]

# Generation settings match app.py so the score reflects what the app produces.
MAX_INPUT_TOKENS = 1024
GENERATION_KWARGS = dict(
    max_length=128,
    min_length=30,
    num_beams=1,
    no_repeat_ngram_size=2,
    repetition_penalty=2.0,
    early_stopping=True,
)


def lead3(text: str) -> str:
    """First three sentences — the standard extractive baseline."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:3]) + "." if sentences else ""


def generate_with_model(texts, model_path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    outputs = []
    for text in texts:
        inputs = tokenizer(
            text, max_length=MAX_INPUT_TOKENS, truncation=True, return_tensors="pt"
        )
        ids = model.generate(input_ids=inputs["input_ids"], **GENERATION_KWARGS)
        outputs.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["lead3", "model"], default="lead3")
    parser.add_argument("--model-path", default="argus_v2_bart")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--summaries-dir", default="data/summaries")
    parser.add_argument("--json", type=Path, help="Also write results to this path.")
    args = parser.parse_args()

    corpus = load_corpus(args.processed_dir, args.summaries_dir)
    print(corpus.describe())
    if not corpus.pairs:
        return 1

    documents = [doc.read_text(encoding="utf-8") for doc, _ in corpus.pairs]
    references = [ref.read_text(encoding="utf-8").strip() for _, ref in corpus.pairs]

    label = f"model ({args.model_path})" if args.system == "model" else "lead-3 baseline"
    print(f"\nScoring: {label}\n")

    if args.system == "model":
        predictions = generate_with_model(documents, args.model_path)
    else:
        predictions = [lead3(text) for text in documents]

    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)

    rows = []
    for (doc, _), prediction, reference in zip(corpus.pairs, predictions, references):
        scores = scorer.score(reference, prediction)
        rows.append(
            {
                "document": doc.stem,
                **{m: round(scores[m].fmeasure, 4) for m in METRICS},
            }
        )

    name_width = max(len(r["document"]) for r in rows)
    name_width = min(name_width, 52)
    header = f"{'document'.ljust(name_width)}  " + "  ".join(m.rjust(8) for m in METRICS)
    print(header)
    print("-" * len(header))
    for row in rows:
        name = row["document"]
        if len(name) > name_width:
            name = name[: name_width - 1] + "…"
        print(
            name.ljust(name_width)
            + "  "
            + "  ".join(f"{row[m]:.4f}".rjust(8) for m in METRICS)
        )

    means = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
    print("-" * len(header))
    print(
        "mean".ljust(name_width)
        + "  "
        + "  ".join(f"{means[m]:.4f}".rjust(8) for m in METRICS)
    )
    print(f"\nn = {len(rows)} document(s).")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"system": label, "per_document": rows, "mean": means, "n": len(rows)},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
