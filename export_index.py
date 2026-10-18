"""Export indexed learning passages and vectors, never credentials or chat history."""

import gzip
import json
import re
from pathlib import Path

from src.retrieval import MODEL_NAME, SNAPSHOT, _get_collection


def export_snapshot(destination=SNAPSHOT):
    collection = _get_collection()
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    count = len(data["ids"])
    if not count:
        raise RuntimeError("Index is empty; run ingest.py first")
    if count > collection._client.get_max_batch_size():
        raise RuntimeError("Snapshot exceeds the atomic import limit; add batched snapshot import before exporting")
    if re.search(r"(?:gsk_|ghp_|github_pat_|oc_sk_)[A-Za-z0-9_-]{20,}", "\n".join(data["documents"])):
        raise RuntimeError("Possible credential in indexed notes; remove it before publishing")
    snapshot = {"embedder": MODEL_NAME, **{key: data[key] for key in ("ids", "documents", "metadatas")}}
    snapshot["embeddings"] = data["embeddings"].tolist()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode("utf-8"), mtime=0))
    temporary.replace(destination)
    return count


if __name__ == "__main__":
    print(f"Exported {export_snapshot()} learning passages to {SNAPSHOT}")
