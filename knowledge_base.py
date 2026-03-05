import os
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
doc_sources = []

def load_documents():
    knowledge_dir = "knowledge"

    for filename in os.listdir(knowledge_dir):
        path = os.path.join(knowledge_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

            documents.append(text)
            doc_sources.append(filename)

load_documents()

embeddings = model.encode(documents)

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def search_docs(query, k=3):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        results.append(documents[i])

    return results