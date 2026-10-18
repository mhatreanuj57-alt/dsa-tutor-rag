import gzip
import json
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.chunking import CODE_LANGUAGES, FILE_LANGUAGES, chunk_repo

ROOT = Path(__file__).parent.parent
DB_DIR = ROOT / "db" / "chroma"
SNAPSHOT = ROOT / "data" / "index-snapshot.json.gz"
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
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_NAME, max_length=384)
    return _reranker


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = _client.get_or_create_collection(name="dsa_notes", metadata={"hnsw:space": "cosine"})
        if collection.count() == 0 and SNAPSHOT.is_file():
            with gzip.open(SNAPSHOT, "rt", encoding="utf-8") as file:
                snapshot = json.load(file)
            if snapshot.pop("embedder") != MODEL_NAME:
                raise RuntimeError("Snapshot embedding model does not match this app")
            # shortcut: one atomic upsert; export rejects snapshots exceeding Chroma's batch limit.
            collection.upsert(**snapshot)
        _collection = collection
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
            {
                "text": doc,
                "source": meta["source"],
                "topic": meta["topic"],
                "language": meta.get("language", FILE_LANGUAGES.get(Path(meta["source"]).suffix, "text")),
            }
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        _bm25 = BM25Okapi([_tokenize(d["text"]) for d in _bm25_docs])
    return _bm25


def reset_bm25():
    global _bm25, _bm25_docs
    _bm25 = None
    _bm25_docs = None


def build_index(notes_dir=None, force=False, sources=None):
    """Synchronize the complete corpus. --force remains accepted for older callers."""
    if notes_dir is not None and sources is not None:
        raise ValueError("Choose notes_dir or language sources, not both")
    if sources is not None and (not sources or any(key not in CODE_LANGUAGES for key in sources)):
        raise ValueError("Sources must map cpp, python, or java to notes directories")
    roots = sources if sources is not None else {None: Path(notes_dir or ROOT / "data" / "notes")}
    chunks = []
    for language, root in roots.items():
        root = Path(root)
        if not root.is_dir():
            raise RuntimeError(f"Notes directory not found: {root}")
        source_chunks = chunk_repo(root, language=language, source_prefix=language)
        if not source_chunks:
            raise RuntimeError(f"No supported learning material found in {root}")
        chunks.extend(source_chunks)

    model = _get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match the source chunks")
    collection = _get_collection()
    ids = [f"{c['source']}#{c['chunk_index']}" for c in chunks]
    previous_ids = set(collection.get(include=[])["ids"])

    batch = 500
    for i in range(0, len(chunks), batch):
        end = i + batch
        collection.upsert(
            ids=ids[i:end],
            documents=[c["text"] for c in chunks[i:end]],
            embeddings=embeddings[i:end].tolist(),
            metadatas=[{key: c[key] for key in ("source", "topic", "language")} for c in chunks[i:end]],
        )
    stale_ids = sorted(previous_ids - set(ids))
    for i in range(0, len(stale_ids), batch):
        collection.delete(ids=stale_ids[i : i + batch])
    reset_bm25()
    return len(chunks)


def index_stats():
    collection = _get_collection()
    languages = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        language = meta.get("language", FILE_LANGUAGES.get(Path(meta["source"]).suffix, "text"))
        languages[language] = languages.get(language, 0) + 1
    return {
        "chunks": collection.count(),
        "languages": languages,
        "embedder": MODEL_NAME,
        "reranker": RERANKER_NAME.split("/")[-1],
    }


def _vector_search(query: str, k: int, language=None):
    collection = _get_collection()
    embedding = _get_model().encode([query]).tolist()
    options = {"where": {"language": language}} if language else {}
    result = collection.query(query_embeddings=embedding, n_results=k, **options)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "topic": meta["topic"],
                "language": meta.get("language", FILE_LANGUAGES.get(Path(meta["source"]).suffix, "text")),
                "distance": round(float(dist), 4),
            }
        )
    return hits


def _rrf_fuse(vector_hits, bm25_hits, k: int = 60):
    scores, pool = {}, {}
    for hits in (vector_hits, bm25_hits):
        for rank, hit in enumerate(hits):
            key = (hit["source"], hit["text"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            pool.setdefault(key, hit.copy())
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [pool[key] for key, _ in ranked]


def hybrid_search(query: str, k: int = 5, candidates: int = 20, rerank: bool = True, language=None):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if language is not None and language not in CODE_LANGUAGES:
        raise ValueError("Language must be cpp, python, or java")
    if k < 1 or candidates < 1:
        raise ValueError("Result counts must be positive")

    if _get_collection().count() == 0:
        raise RuntimeError("Index is empty - run ingest.py first")

    bm25 = _get_bm25()
    eligible = [i for i, doc in enumerate(_bm25_docs) if language is None or doc["language"] == language]
    if not eligible:
        return []
    candidates = min(max(k, candidates), len(eligible))
    vector_hits = _vector_search(query, candidates, language=language)
    bm25_scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(eligible, key=lambda i: -bm25_scores[i])[:candidates]
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


def search(query: str, k: int = 5, language=None):
    return hybrid_search(query, k=k, language=language)
