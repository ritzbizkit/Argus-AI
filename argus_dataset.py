import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from corpus import load_corpus


class SecurityComplianceDataset(Dataset):
    """Document/summary pairs for fine-tuning the synthesis model.

    Pairing is delegated to `corpus.load_corpus`, which also reports files that
    are unpaired on either side — a summary with no processed document usually
    means `extract_text.py` produced no text for that PDF, which quietly shrinks
    the training set.
    """

    def __init__(
        self,
        processed_dir,
        summaries_dir,
        model_name="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.corpus = load_corpus(processed_dir, summaries_dir)
        self.pairs = self.corpus.pairs

        if not self.pairs:
            raise RuntimeError(
                f"No document/summary pairs found between {processed_dir} "
                f"and {summaries_dir}."
            )

    def describe(self) -> str:
        return self.corpus.describe()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]

        source_text = input_path.read_text(encoding="utf-8")
        target_text = target_path.read_text(encoding="utf-8")

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encoding["input_ids"].flatten(),
            "attention_mask": source_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten(),
        }
