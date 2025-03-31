from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = "Mens bracelets are made from leather, metal, or beads. They gained popularity in the 2000s."
question = "What materials are used to make mens bracelets?"

result = qa_pipeline(question=question, context=context)

print(result['answer'])  # Output: leather, metal, or beads
