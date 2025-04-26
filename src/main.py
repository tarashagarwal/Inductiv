from transformers import pipeline


classifier = pipeline(
    "text-classification",
    model="tarashagarwal/inductiv-binary-classifier",  # replace with your actual repo name
    tokenizer="tarashagarwal/inductiv-binary-classifier"
)

def test_query(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    print(f"Prediction: {label}, Confidence: {score}")
    print("asasas:" ,float(result['score']) > 0.50)
    return result["label"] if float(result["score"]) > 0.50  else  reverseLabel(result["label"])

def reverseLabel(label):
    return "LABEL_0" if label == "LABEL_1" else "LABEL_1"


# test_query("Who teaches Machine Learning this fall?")
# test_query("What's the weather like in Paris?")
# test_query("Guruparan, A;Mukhopadhyay si the instructor for Fall for BADM2200?")
# test_query("Asish K for Fall for BADM2200? in summer")
# test_query("What is the univeristy ranking?")
# test_query("What are some some hot topics?")
# test_query("Is BADM576 taught this semester?")

while True:
    prompt = input("Hpw can I help?\n")
    if prompt == "stop":
        exit()
    else:
        print(test_query(prompt))
