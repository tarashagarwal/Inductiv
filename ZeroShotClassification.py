from transformers import pipeline

classifier = pipeline("zero-shot-classification")

sentence = "The Great Grand Father of the man who freeded the nation was a traitor who earned millions and set up the most successful business of the decade"
labels = ["Freedom", "Finance", "Sports", "Politics", "History"]

result = classifier(sentence, labels)
print(result)
