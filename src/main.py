from transformers import pipeline


# classifier = pipeline(
#     "text-classification",
#     model="tarashagarwal/inductiv-binary-classifier",  # replace with your actual repo name
#     tokenizer="tarashagarwal/inductiv-binary-classifier"
# )

# def test_query(text):
#     result = classifier(text)[0]
#     label = result["label"]
#     score = result["score"]
#     print(f"Prediction: {label}, Confidence: {score}")
#     print("asasas:" ,float(result['score']) > 0.50)
#     return result["label"] if float(result["score"]) > 0.50  else  reverseLabel(result["label"])

# def reverseLabel(label):
#     return "LABEL_0" if label == "LABEL_1" else "LABEL_1"


# # test_query("Who teaches Machine Learning this fall?")
# # test_query("What's the weather like in Paris?")
# # test_query("Guruparan, A;Mukhopadhyay si the instructor for Fall for BADM2200?")
# # test_query("Asish K for Fall for BADM2200? in summer")
# # test_query("What is the univeristy ranking?")
# # test_query("What are some some hot topics?")
# # test_query("Is BADM576 taught this semester?")

# while True:
#     prompt = input("How can I help?\n")
#     if prompt == "stop":
#         exit()
#     else:
#         print(test_query(prompt))


# 1. Install the required libraries
# !pip install transformers

# 2. Define your model repo
# (Replace below with your real repo name)
# repo_id = "cahya/NusaBert-ner-v1.3" 

# 3. Load the model + tokenizer directly from Hugging Face
# ner_pipeline = pipeline(
#     "ner",
#     model=repo_id,
#     tokenizer=repo_id,
#     aggregation_strategy="simple"   # Optional: Combines multi-token words nicely
# )

# # 4. Test with a sentence
# sentence = "During Spring 2024, I enrolled for AAS100, also known as Intro Asian American Studies, under the guidance of Guruparan, A, earning 3 credits upon completion in Fall 2023."

# results = ner_pipeline(sentence)

# # 5. Display results
# print("\nNER Results:")
# for entity in results:
#     print(f"{entity['word']} ({entity['entity_group']}) - Confidence: {entity['score']:.4f}")


from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")

text = """
During Spring 2024, I enrolled for AAS100, also known as Intro Asian American Studies, under the guidance of Guruparan, A, earning 3 credits upon completion in Fall 2023.
"""

labels = ["instructor", "time", "course", "code", "sports"]

entities = model.predict_entities(text, labels, threshold=0.3)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
