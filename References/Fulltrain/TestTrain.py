# test_mlm_model.py
# ✅ Script to test the BERT MLM model trained from scratch

from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

# ----------------------------
# 1. Load Model and Tokenizer
# ----------------------------
MODEL_PATH = "./bert-from-scratch"

print("\n📦 Loading model and tokenizer from:", MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

# ----------------------------
# 2. Create Fill-Mask Pipeline
# ----------------------------
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# ----------------------------
# 3. Test Sentences
# ----------------------------
test_sentences = [
    "Mens [MASK] with diamonds.",
    "Womens [MASK] for weddings.",
    "Gold [MASK] are popular.",
    "Silver [MASK] are stylish.",
    "Temple [MASK] is traditional."
]

print("\n🧪 Running Predictions:")
for sentence in test_sentences:
    print(f"\n🔸 Input: {sentence}")
    outputs = fill_mask(sentence)
    for o in outputs:
        print(f" → {o['sequence']}  (score: {o['score']:.4f})")