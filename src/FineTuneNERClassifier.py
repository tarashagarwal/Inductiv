# Install necessary libraries if not already installed
# !pip install transformers datasets huggingface_hub python-dotenv

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os

# 1. Load environment variables (your Hugging Face token)
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise ValueError("❌ Hugging Face Token not found in environment variables. Set HUGGINGFACE_TOKEN first!")

# Login to Hugging Face Hub
login(token=hf_token)

# 2. Load the dataset from Hugging Face
dataset = load_dataset("tarashagarwal/inductiv-ner-course-professor-semester-classifier")
print("\nDataset structure:", dataset)
print("\nDataset features:", dataset["train"].features)
print("\nSample example:", dataset["train"][0])

# 3. Dynamically create the labels from dataset
# Assuming the "labels" field is present
unique_labels = set()
for example in dataset["train"]:
    unique_labels.update(example["labels"])    # If your dataset field is named differently (e.g., "ner_tags"), change it here

labels = sorted(list(unique_labels))
print("\nLabels:", labels)

# Create label2id and id2label mappings
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# 4. Load Pretrained Tokenizer and Model
checkpoint = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 5. Tokenization and label alignment
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=32,
    )

    labels_aligned = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels_aligned.append(-100)
        elif word_idx != previous_word_idx:
            labels_aligned.append(label2id[example["labels"][word_idx]])  # Adjust here if field name is different
        else:
            labels_aligned.append(label2id[example["labels"][word_idx]])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels_aligned
    return tokenized_inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

# 6. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-ner-custom",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id="tarashagarwal/inductiv-ner-course-professor-semester-model",  # <-- Upload to NEW model repo, not dataset
    hub_strategy="every_save",
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],  # Later you can split into train/validation
    tokenizer=tokenizer,
)

# 8. Fine-tune and Push to Hugging Face Hub
trainer.train()
trainer.push_to_hub()

print("\n✅ Model pushed successfully to Hugging Face Hub!")
