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
import os

from dotenv import load_dotenv

load_dotenv()


# 1. Login to Hugging Face Hub
login(token=os.getenv("HUGGINGFACE_TOKEN")) # paste your token from https://huggingface.co/settings/tokens

# 2. Load Pretrained Model and Tokenizer
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 3. Example Dataset for Fine-Tuning
# with open('../TrainData/InductivClassifierTrain.json', 'r') as f:
#     examples = json.load(f)

dataset = load_dataset("tarashagarwal/inductiv-binary-classifier")

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset["train"].map(tokenize_fn, batched=True)


train_test_split = tokenized_dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


# 4. Tokenization Function
# def preprocess_function(examples):
#     return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)

# tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 5. Set up Training Arguments
repo_name = "tarashagarwal/inductive-classifier"

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,              # smaller LR
    per_device_train_batch_size=8,   # bigger batch
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id="tarashagarwal/inductiv-binary-classifier",
    report_to="none",
    optim="adamw_torch_fused",
    fp16=True,                       # enable mixed precision
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# 6. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 7. Fine-tune and Push
print("Starting Training")
trainer.train()
print("Training Complete. Now pushing to the repository")
trainer.push_to_hub()