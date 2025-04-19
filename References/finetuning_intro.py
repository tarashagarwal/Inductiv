from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)