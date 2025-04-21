# 📦 Install dependencies first
# pip install transformers datasets accelerate huggingface_hub

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from huggingface_hub import login
import json

# 1. Login to Hugging Face Hub
login(token='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') # paste your token from https://huggingface.co/settings/tokens

# 2. Load Pretrained Model and Tokenizer
model_checkpoint = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3  # You can change this based on your task (e.g., "general", "course", "other")
)

# 3. Example Dataset for Fine-Tuning
with open('../TrainData/InductivClassifierTrain.json', 'r') as f:
    examples = json.load(f)

dataset = load_dataset("tarashagarwal/inductiv-binary-classifier")

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset["train"].map(tokenize_fn, batched=True)

# 4. Tokenization Function
# def preprocess_function(examples):
#     return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)

# tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 5. Set up Training Arguments
repo_name = "tarashagarwal/inductive-classifier"

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id=repo_name,
    report_to="none",
    optim="adamw_torch_fused",  # Accelerate optimized
)

# 6. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 7. Fine-tune and Push
print("Starting Training")
trainer.train()
print("Training Complete. Now pushing to the repository")
trainer.push_to_hub()