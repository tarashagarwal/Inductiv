from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = ["general", "course-related"]

query = "Hello, Did not see you in class today?"

result = classifier(query, candidate_labels=labels)
print(result)


# 1. Load the Roberta Large NER Model
# ner_pipeline = pipeline(
#     "ner",
#     model="dbmdz/bert-large-cased-finetuned-conll03-english",
#     aggregation_strategy="simple"  # This merges multi-token entities nicely
# )

# # 2. Define a user query
# query = "Professor John Smith teaches Data Mining at Stanford University."

# # 3. Run NER
# entities = ner_pipeline(query)

# # 4. Print the detected entities
# for entity in entities:
#     print(f"Entity: {entity['word']} | Label: {entity['entity_group']} | Score: {entity['score']:.2f}")


classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

candidate_labels = ["course-related", "general", "sports", "weather"]
sequence_to_classify = "Can you give description for CSCI2200?"

result = classifier(sequence_to_classify, candidate_labels)
print(result)