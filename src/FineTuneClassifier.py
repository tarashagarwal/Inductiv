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

# 1. Login to Hugging Face Hub
login()  # paste your token from https://huggingface.co/settings/tokens

# 2. Load Pretrained Model and Tokenizer
model_checkpoint = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3  # You can change this based on your task (e.g., "general", "course", "other")
)

# 3. Example Dataset for Fine-Tuning
examples = [
    {"text": "What is the weather today?", "label": 0},  # 0 = General
    {"text": "Which courses are taught by Professor Smith?", "label": 1},  # 1 = Course
    {"text": "What are the hobbies of Prof. John?", "label": 2},  # 2 = Interest
]

dataset = Dataset.from_dict({
    "text": [e["text"] for e in examples],
    "label": [e["label"] for e in examples]
})

# 4. Tokenization Function
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

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
trainer.train()
trainer.push_to_hub()
