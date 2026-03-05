import os
import faiss
import threading
from sentence_transformers import SentenceTransformer

# Globals
model = None
index = None
documents = []
doc_sources = []
_ready = False
_lock = threading.Lock()

# Ensure paths work both locally and on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")


def _initialize():
    """
    Lazy-load the embedding model and build the FAISS index.
    This prevents Render startup from timing out.
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

        # Load documents
        for filename in os.listdir(KNOWLEDGE_DIR):
            path = os.path.join(KNOWLEDGE_DIR, filename)

            if not filename.endswith(".txt"):
                continue

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

                documents.append(text)
                doc_sources.append(filename)

        if not documents:
            raise ValueError("No documents found in knowledge directory.")

        # Create embeddings
        embeddings = model.encode(documents)

        dimension = len(embeddings[0])

        # Build FAISS index
        idx = faiss.IndexFlatL2(dimension)
        idx.add(embeddings)

        index = idx
        _ready = True

        print("Knowledge base initialized successfully.")


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