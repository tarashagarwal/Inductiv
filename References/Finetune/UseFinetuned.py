from transformers import pipeline

fill_mask = pipeline("fill-mask", model="./bert-jewellery-model", tokenizer="./bert-jewellery-model")

results = fill_mask("Mens [MASK]", top_k=10)

for res in results:
    print(res['sequence'], "(score:", round(res['score'], 4), ")")
