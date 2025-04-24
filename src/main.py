from transformers import pipeline

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# labels = ["general", "course-related"]

# query = "Hello, Did not see you in class today?"

# result = classifier(query, candidate_labels=labels)
# print(result)


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


# classifier = pipeline("zero-shot-classification", model="tarashagarwal/inductiv-binary-classifier")

# candidate_labels = ["course-related", "general", "sports", "weather"]
# sequence_to_classify = "Can you give description for CSCI2200?"

# result = classifier(sequence_to_classify, candidate_labels)
# print(result)


classifier = pipeline(
    "text-classification",
    model="tarashagarwal/inductiv-binary-classifier",  # replace with your actual repo name
    tokenizer="tarashagarwal/inductiv-binary-classifier"
)

def test_query(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    print(f"Prediction: {label}, Confidence: {score:.2f}")

test_query("Who teaches Machine Learning this fall?")
test_query("What's the weather like in Paris?")
test_query("Guruparan, A;Mukhopadhyay si the instructor for Fall for BADM2200?")
test_query("Asish K for Fall for BADM2200? in summer")
test_query("What is the univeristy ranking?")
test_query("What are some some hot topics?")
test_query("Is BADM576 taught this semester?")


