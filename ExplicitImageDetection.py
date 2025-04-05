from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch

model_id = "Falconsai/nsfw_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_id)
extractor = AutoFeatureExtractor.from_pretrained(model_id)

image = Image.open("explicit.jpg").convert("RGB")
inputs = extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

# Get label
pred_id = probs.argmax().item()
label = model.config.id2label[pred_id]

print(f"🔥 Predicted Class: {label}")
print("📊 Probabilities:")
for i, prob in enumerate(probs[0]):
    print(f"{model.config.id2label[i]}: {prob.item():.4f}")
