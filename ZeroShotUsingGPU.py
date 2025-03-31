import time
from transformers import pipeline

use_gpu = True
device = 0 if use_gpu else -1

classifier = pipeline("sentiment-analysis", device=device)

# Create 100 sentences
texts = ["I love this product!"] * 100

start_time = time.time()
results = classifier(texts)
end_time = time.time()

print(f"⏱️ Time on {'GPU' if use_gpu else 'CPU'}: {end_time - start_time:.2f} seconds")
