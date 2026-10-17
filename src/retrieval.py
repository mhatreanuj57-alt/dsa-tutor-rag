from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.chunking import chunk_repo

ROOT = Path(__file__).parent.parent
DB_DIR = ROOT / "db" / "chroma"
MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_model = None
_reranker = None
_client = None
_collection = None
_bm25 = None
_bm25_docs = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_NAME, max_length=384)
    return _reranker


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = _client.get_or_create_collection(
            name="dsa_notes", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _tokenize(text: str):
    return text.lower().replace("\n", " ").split()


def _get_bm25():
    global _bm25, _bm25_docs
    if _bm25 is None:
        collection = _get_collection()
        if collection.count() == 0:
            raise RuntimeError("Index is empty - run ingest.py first")
        data = collection.get(include=["documents", "metadatas"])
        _bm25_docs = [
            {"text": doc, "source": meta["source"], "topic": meta["topic"]}
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        _bm25 = BM25Okapi([_tokenize(d["text"]) for d in _bm25_docs])
    return _bm25


def reset_bm25():
    global _bm25
    _bm25 = None


def build_index(notes_dir=None, force=False):
    notes_dir = Path(notes_dir or ROOT / "data" / "notes")
    if not notes_dir.exists():
        raise RuntimeError(f"Notes directory not found: {notes_dir}")

    collection = _get_collection()
    if force:
        _client.delete_collection("dsa_notes")
        _collection = _client.get_or_create_collection(
            name="dsa_notes", metadata={"hnsw:space": "cosine"}
        )
        collection = _collection

    chunks = chunk_repo(notes_dir)
    if not chunks:
        raise RuntimeError(f"No files found in {notes_dir}")

    model = _get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    batch = 500
    for i in range(0, len(chunks), batch):
        end = i + batch
        collection.upsert(
            ids=[f"{c['source']}#{c['chunk_index']}" for c in chunks[i:end]],
            documents=[c["text"] for c in chunks[i:end]],
            embeddings=embeddings[i:end].tolist(),
            metadatas=[{"source": c["source"], "topic": c["topic"]} for c in chunks[i:end]],
        )
    reset_bm25()
    return len(chunks)


def index_stats():
    return {
        "chunks": _get_collection().count(),
        "embedder": MODEL_NAME,
        "reranker": RERANKER_NAME.split("/")[-1],
    }


def _vector_search(query: str, k: int):
    collection = _get_collection()
    embedding = _get_model().encode([query]).tolist()
    result = collection.query(query_embeddings=embedding, n_results=k)
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": doc, "source": meta["source"], "topic": meta["topic"],
                     "distance": round(float(dist), 4)})
    return hits


def _rrf_fuse(vector_hits, bm25_hits, k: int = 60):
    scores = {}
    for rank, hit in enumerate(vector_hits):
        key = hit["source"] + "|" + hit["text"][:80]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        hit["_key"] = key
    for rank, hit in enumerate(bm25_hits):
        key = hit["source"] + "|" + hit["text"][:80]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        hit["_key"] = key
    pool = {h["_key"]: h for h in vector_hits + bm25_hits}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [pool[key] for key, _ in ranked]


def hybrid_search(query: str, k: int = 5, candidates: int = 20, rerank: bool = True):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if _get_collection().count() == 0:
        raise RuntimeError("Index is empty - run ingest.py first")

    vector_hits = _vector_search(query, candidates)
    bm25 = _get_bm25()
    bm25_scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:candidates]
    bm25_hits = [_bm25_docs[i] for i in top_idx if bm25_scores[i] > 0]

    fused = _rrf_fuse(vector_hits, bm25_hits)[:candidates]

    if rerank and fused:
        reranker = _get_reranker()
        pairs = [(query, h["text"]) for h in fused]
        scores = reranker.predict(pairs)
        for hit, score in zip(fused, scores):
            hit["relevance"] = round(float(score), 4)
        fused = sorted(fused, key=lambda h: -h["relevance"])

    return fused[:k]


def search(query: str, k: int = 5):
    return hybrid_search(query, k=k)
