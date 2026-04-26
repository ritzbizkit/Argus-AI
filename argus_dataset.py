import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SecurityComplianceDataset(Dataset):
    def __init__(self, processed_dir, summaries_dir, model_name="facebook/bart-large-cnn", max_input_length=1024, max_target_length=128):
        self.processed_dir = processed_dir
        self.summaries_dir = summaries_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.filenames = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_path = os.path.join(self.processed_dir, filename)
        
        summary_filename = filename.replace('.txt', '_summary.txt')
        target_path = os.path.join(self.summaries_dir, summary_filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            source_text = f.read() # Notice we removed the "summarize:" prefix!
            
        with open(target_path, 'r', encoding='utf-8') as f:
            target_text = f.read()

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": source_encoding["input_ids"].flatten(),
            "attention_mask": source_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten()
        }