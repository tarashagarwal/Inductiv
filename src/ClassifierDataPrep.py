from datasets import Dataset
import pandas as pd
from huggingface_hub import login
df = pd.read_json("../TrainData/InductivClassifierTrain.json")

dataset = Dataset.from_pandas(df)

login(token='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')

dataset.push_to_hub("tarashagarwal/inductiv-binary-classifier")