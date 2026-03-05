import os
import json
import time
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
from openai import OpenAI

client = OpenAI()

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CACHE_DIR = os.path.join(BASE_DIR, ".kb_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "embeddings.json")


# -------- Config --------
EMBED_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 4

# Chunking (tweak if you want)
CHUNK_TARGET_CHARS = 1400   # ~250-350 words depending on text
CHUNK_OVERLAP_CHARS = 250   # overlap to keep context across boundaries


# -------- In-memory store --------
_chunks: List[Dict[str, Any]] = []   # each: {id, source, text}
_embeddings: np.ndarray | None = None


def _ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _file_hash(path: str) -> str:
    """Hash file contents so we can invalidate cache when docs change."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _split_into_chunks(text: str) -> List[str]:
    """
    Split text into overlapping chunks by character length.
    Simple + reliable for CV/dissertation text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = "\n".join([ln for ln in text.split("\n") if ln.strip()])

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_TARGET_CHARS, n)

        # Try not to cut mid-sentence if possible
        if end < n:
            cut = text.rfind(".", start, end)
            if cut != -1 and (end - cut) < 200:
                end = cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - CHUNK_OVERLAP_CHARS)

    return chunks


def _load_documents() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load .txt docs from /knowledge and return chunk list + metadata for cache.
    """
    if not os.path.isdir(KNOWLEDGE_DIR):
        raise FileNotFoundError(f"Knowledge directory not found: {KNOWLEDGE_DIR}")

    files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".txt")]
    if not files:
        raise ValueError(f"No .txt files found in {KNOWLEDGE_DIR}")

    meta = {"files": {}, "created_at": int(time.time())}
    chunks: List[Dict[str, Any]] = []

    for filename in sorted(files):
        path = os.path.join(KNOWLEDGE_DIR, filename)
        meta["files"][filename] = _file_hash(path)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        for i, chunk_text in enumerate(_split_into_chunks(text)):
            chunks.append({
                "id": f"{filename}::chunk{i}",
                "source": filename,
                "text": chunk_text
            })

    return chunks, meta


def _read_cache() -> Dict[str, Any] | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(payload: Dict[str, Any]):
    _ensure_dirs()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _cache_matches(cache: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    """
    Cache is valid if the set of files and hashes match.
    """
    if not cache or "meta" not in cache:
        return False
    cached_files = cache["meta"].get("files", {})
    return cached_files == meta.get("files", {})


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings in batches.
    """
    # OpenAI embeddings endpoint supports batching via input=list[str]
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


def _build_or_load_index():
    """
    Build embeddings (or load cached) into memory.
    """
    global _chunks, _embeddings

    chunks, meta = _load_documents()
    cache = _read_cache()

    if cache and _cache_matches(cache, meta):
        # Load from cache
        _chunks = cache["chunks"]
        _embeddings = np.array(cache["embeddings"], dtype=np.float32)
        return

    # Build new embeddings
    texts = [c["text"] for c in chunks]

    # Batch size—safe + fast
    batch_size = 64
    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = _embed_texts(batch)
        all_vecs.extend(vecs)

    _chunks = chunks
    _embeddings = np.array(all_vecs, dtype=np.float32)

    # Write cache
    payload = {
        "meta": meta,
        "chunks": _chunks,
        "embeddings": _embeddings.tolist()
    }
    _write_cache(payload)


def _cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity of query to each row in matrix.
    """
    # Normalize
    mat = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    return mat @ q


def search_docs(query: str, k: int = DEFAULT_TOP_K) -> List[str]:
    """
    Return top-k relevant chunk texts for the query.
    """
    global _embeddings, _chunks

    if _embeddings is None or not _chunks:
        _build_or_load_index()

    q_vec = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding
    q_vec = np.array(q_vec, dtype=np.float32)

    scores = _cosine_scores(_embeddings, q_vec)
    top_idx = np.argsort(scores)[::-1][:k]

    return [_chunks[i]["text"] for i in top_idx]


def search_docs_with_sources(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, str]]:
    """
    Same as search_docs, but includes sources and chunk ids (useful for debugging).
    """
    global _embeddings, _chunks

    if _embeddings is None or not _chunks:
        _build_or_load_index()

    q_vec = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding
    q_vec = np.array(q_vec, dtype=np.float32)

    scores = _cosine_scores(_embeddings, q_vec)
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_idx:
        results.append({
            "id": _chunks[i]["id"],
            "source": _chunks[i]["source"],
            "text": _chunks[i]["text"]
        })
    return results