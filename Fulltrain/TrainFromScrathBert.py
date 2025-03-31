# =======================================
# TRAIN BERT FROM SCRATCH (Masked LM)
# =======================================

from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

# -------------------------
# 1. Define Config for BERT (smaller version)
# -------------------------
config = BertConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
)

model = BertForMaskedLM(config)

# -------------------------
# 2. Load Tokenizer (Assume you trained one already)
# -------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

print("Tokenizer vocab size:", tokenizer.vocab_size)

# -------------------------
# 3. Load Dataset
# -------------------------
dataset = load_dataset("text", data_files={"train": "my_data/part1.txt"})

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# -------------------------
# 4. Data Collator (MLM)
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# -------------------------
# 5. Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./bert-from-scratch",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=50,
)

# -------------------------
# 6. Trainer Setup
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# -------------------------
# 7. Train
# -------------------------
trainer.train()

# -------------------------
# 8. Save Model and Tokenizer
# -------------------------
model.save_pretrained("./bert-from-scratch")
tokenizer.save_pretrained("./bert-from-scratch")
