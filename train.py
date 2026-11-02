"""Fine-tune the synthesis model on the regulatory document/summary corpus.

Usage:
    python train.py [--epochs N] [--processed-dir DIR] [--allow-tiny-dataset]

The corpus in this repository is very small. Read the warning the script prints
before treating any number it produces as a result.
"""

import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "./argus_v2_bart"

# Below this many pairs, a held-out split is too small for the eval loss to mean
# anything. The script refuses to run rather than produce a misleading number.
MIN_PAIRS_FOR_MEANINGFUL_EVAL = 20


def build_training_arguments(**kwargs):
    """Construct training arguments across transformers versions.

    `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.46.
    """
    import inspect

    from transformers import Seq2SeqTrainingArguments

    strategy = kwargs.pop("eval_strategy")
    field = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    kwargs[field] = strategy
    return Seq2SeqTrainingArguments(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--processed-dir",
        default="data/sections",
        help="Directory of source documents (default: the 35-section corpus).",
    )
    parser.add_argument("--summaries-dir", default="data/summaries")
    parser.add_argument(
        "--allow-tiny-dataset",
        action="store_true",
        help="Train anyway when there are too few pairs for a meaningful eval split.",
    )
    args = parser.parse_args()

    # Imported here so --help works without the model stack installed.
    from torch.utils.data import Subset
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer

    from argus_dataset import SecurityComplianceDataset

    dataset = SecurityComplianceDataset(
        args.processed_dir, args.summaries_dir, model_name=MODEL_NAME
    )
    print(dataset.describe())

    n = len(dataset)
    if n < MIN_PAIRS_FOR_MEANINGFUL_EVAL and not args.allow_tiny_dataset:
        print(
            f"\nSTOPPING: {n} document/summary pair(s) is not enough to hold out a\n"
            f"meaningful evaluation set (want at least {MIN_PAIRS_FOR_MEANINGFUL_EVAL}).\n"
            "Fine-tuning a model this size on a corpus this small will memorise it,\n"
            "and any eval loss you get will say nothing about generalisation.\n\n"
            "Either add more document/summary pairs, or re-run with\n"
            "--allow-tiny-dataset if you understand the result is not evidence of\n"
            "model quality.",
        )
        return 1

    # Hold out the last 20% (at least one example). Fixed rather than random so
    # repeated runs are comparable.
    n_eval = max(1, n // 5)
    train_dataset = Subset(dataset, list(range(n - n_eval)))
    eval_dataset = Subset(dataset, list(range(n - n_eval, n)))
    print(f"Training on {len(train_dataset)}, evaluating on {len(eval_dataset)}.")

    if n < MIN_PAIRS_FOR_MEANINGFUL_EVAL:
        print(
            "\nWARNING: running on a tiny dataset at your request. The eval loss\n"
            "below is not evidence that the model generalises.\n"
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    training_args = build_training_arguments(
        output_dir="./argus-bart-checkpoints",
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=20,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # batch of 1 to fit in 16GB VRAM
        gradient_accumulation_steps=2,  # effective batch size of 2
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        logging_steps=5,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    dataset.tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
