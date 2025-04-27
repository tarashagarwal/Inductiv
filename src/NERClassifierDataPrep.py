# 1. Install required libraries
# !pip install datasets huggingface_hub

from datasets import load_dataset, Dataset
from huggingface_hub import login
import json
import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


# 2. Authenticate with Hugging Face
# Replace with your actual Hugging Face token (you can create it from https://huggingface.co/settings/tokens)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)

# 3. Load your custom dataset
# Assuming your data is saved as JSONL file at "data/ner_dataset.jsonl"
df = pd.read_json("../TrainData/InductiveNERClassifierTrain.json")

dataset = Dataset.from_pandas(df)

# 4. Push to Hugging Face Hub

# Set your repo name (e.g., username/dataset_name)
repo_id = "dslim/bert-base-NER"

# Push the dataset
dataset.push_to_hub(repo_id)

print(f"Dataset pushed successfully to https://huggingface.co/datasets/{repo_id}")
