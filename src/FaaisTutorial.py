# Import libraries
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Step 1: Setup model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small fast model

dimension = 384  # Dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Step 2: Sentences to store
sentences = [
    "Professor John teaches AI in Fall 2025",
    "Mary will teach Data Science in Spring 2025",
    "Machine Learning course is offered by Dr. Smith",
    "Introduction to Cybersecurity by Prof. Alex"
]

# Step 3: Encode sentences to vectors
embeddings = model.encode(sentences)

# Step 4: Add embeddings to FAISS index
index.add(np.array(embeddings))

# (Optional) Save sentence metadata separately
with open('sentences.pkl', 'wb') as f:
    pickle.dump(sentences, f)

# Step 5: Save FAISS index to disk
faiss.write_index(index, 'courses.index')

print("✅ Saved index and sentences!")

# --- Now later we can load everything back ---

# Step 6: Load FAISS index and metadata
index = faiss.read_index('courses.index')

with open('sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

print("✅ Loaded index and sentences!")

# Step 7: Search with a query
query = "Suggest me a professor who is teaching Machine lEarning in 2025?"
query_embedding = model.encode([query])

# Search top 3 closest sentences
D, I = index.search(np.array(query_embedding), k=3)

# print("\n🔎 Top matching sentences:")
# for idx in I[0]:
#     print("-", sentences[idx])


from sklearn.metrics.pairwise import cosine_similarity

# Search top 3 matches
D, I = index.search(np.array(query_embedding), k=3)

print("\n🔎 Top matching sentences with match %:")
for idx in I[0]:
    matched_sentence = sentences[idx]
    
    # Get vector of matched sentence
    matched_vector = model.encode([matched_sentence])

    # Compute cosine similarity
    similarity = cosine_similarity(query_embedding, matched_vector)[0][0]
    
    # Convert to percentage
    match_percent = round(similarity * 100, 2)

    print(f"- {matched_sentence}  ({match_percent}% match)")
