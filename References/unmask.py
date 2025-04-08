from transformers import pipeline

# Load the fill-mask pipeline with BERT
unmasker = pipeline("fill-mask", model="bert-base-uncased")

# Sentence with a masked word
sentence = "Mens [MASK]"

# Generate predictions
results = unmasker(sentence, top_k=10)

# Show top predictions
for i, r in enumerate(results):
    print(f"{i+1}. {r['sequence']}  (score: {r['score']:.4f})")
