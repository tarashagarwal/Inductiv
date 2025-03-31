from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Step 1: Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Step 2: Load your text data (1 sentence per line)
dataset = load_dataset("text", data_files={"train": "jewellery.txt"})

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Random masking during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./bert-jewellery-model",
    evaluation_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

import torch
print("✅ CUDA available:", torch.cuda.is_available())
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# Step 7: Start training
trainer.train()

# Step 8: Save model and tokenizer
trainer.save_model("./bert-jewellery-model")
tokenizer.save_pretrained("./bert-jewellery-model")
