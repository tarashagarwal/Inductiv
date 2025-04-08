from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-hi"  # English → Hindi

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Input text
text = "I am Thirsty."
inputs = tokenizer(text, return_tensors="pt", padding=True)

# Generate translation
translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)

print("🔁 Translated:", output)
