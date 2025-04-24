from datasets import Dataset
import pandas as pd
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

df = pd.read_json("../TrainData/InductivClassifierTrain.json")

dataset = Dataset.from_pandas(df)

login(token=os.getenv("HUGGINGFACE_TOKEN"))

dataset.push_to_hub("tarashagarwal/inductiv-binary-classifier")