import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from argus_dataset import SecurityComplianceDataset

def main():
    print("Initializing Argus AI v2.0 (Heavyweight Pipeline)...")
    model_name = "facebook/bart-large-cnn"
    
    print(f"Loading Base Model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("Loading Dataset...")
    dataset = SecurityComplianceDataset("data/processed", "data/summaries", model_name=model_name)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./argus-bart-checkpoints",
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=10, 
        save_steps=20,
        num_train_epochs=5,
        per_device_train_batch_size=1,     # Lowered to 1 to fit in 16GB VRAM
        gradient_accumulation_steps=2,     # Simulates a batch size of 2
        per_device_eval_batch_size=1,
        learning_rate=2e-5,                # Slower, more stable learning rate for BART
        weight_decay=0.01,
        fp16=True, 
        logging_dir='./logs',
        logging_steps=5,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    
    print("Starting Training Loop. BART is huge, so this will take a bit longer...")
    trainer.train()
    
    print("Training Complete! Saving Argus AI v2.0...")
    model.save_pretrained("./argus_v2_bart")
    dataset.tokenizer.save_pretrained("./argus_v2_bart")
    print("Model saved to ./argus_v2_bart")

if __name__ == "__main__":
    main()