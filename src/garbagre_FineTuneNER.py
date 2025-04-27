# 📦 Install dependencies first
# pip install transformers datasets accelerate huggingface_hub

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
    TrainingArguments, 
    Trainer
)
from huggingface_hub import login

# 1. Login to Hugging Face
login()  # You'll paste your HF token here from https://huggingface.co/settings/tokens

# 2. Load Pretrained Model and Tokenizer
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=5  # Change if your labels are different
)

# 3. Create a Small Example Dataset (You can replace this with your real data)
examples = [
    {"tokens": ["Professor", "John", "teaches", "CS101"], 
     "tags": ["B-PROFESSOR", "I-PROFESSOR", "O", "B-COURSE"]},
    {"tokens": ["Prof", "Alice", "teaches", "Machine", "Learning"], 
     "tags": ["B-PROFESSOR", "I-PROFESSOR", "O", "B-COURSE", "I-COURSE"]}
]
dataset = Dataset.from_dict({
    "tokens": [e["tokens"] for e in examples],
    "tags": [e["tags"] for e in examples]
})

# 4. Define your Label Mapping
label_list = ["O", "B-PROFESSOR", "I-PROFESSOR", "B-COURSE", "I-COURSE"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# 5. Tokenize and Align Labels
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["tags"][word_idx]])
        else:
            labels.append(label2id[example["tags"][word_idx]])
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

# 6. Define Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 7. Set TrainingArguments (using Accelerate backend)
repo_name = "tarashagarwal/inductiv-"

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
    report_to="none",  # Don't report to wandb etc
    optim="adamw_torch_fused",  # Accelerate optimized optimizer
)

# 8. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 9. Train and Push
trainer.train()
trainer.push_to_hub()
