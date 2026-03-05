import os
import faiss
import threading
from sentence_transformers import SentenceTransformer

# Global variables (initially empty)
model = None
index = None
documents = []
doc_sources = []
_ready = False
_lock = threading.Lock()


def _initialize():
    """
    Load model, documents, and build FAISS index.
    Runs only once (lazy initialization).
    """
    global model, index, documents, doc_sources, _ready

    if _ready:
        return

    with _lock:
        if _ready:
            return

        print("Initializing knowledge base...")

        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        knowledge_dir = "knowledge"

        for filename in os.listdir(knowledge_dir):
            path = os.path.join(knowledge_dir, filename)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

                documents.append(text)
                doc_sources.append(filename)

        # Create embeddings
        embeddings = model.encode(documents)

        # Build FAISS index
        dimension = len(embeddings[0])
        idx = faiss.IndexFlatL2(dimension)
        idx.add(embeddings)

        index = idx
        _ready = True

        print("Knowledge base ready.")


def search_docs(query, k=3):
    """
    Search knowledge base for relevant documents.
    """
    _initialize()

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        results.append(documents[i])

    return results