from pathlib import Path
from io import StringIO

import numpy as np
import pytest

from src import chunking, llm, retrieval


def test_environment_template_loads_the_chat_client(monkeypatch):
    from dotenv import dotenv_values
    from dotenv.parser import parse_stream

    template = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    assert not [item.original.line for item in parse_stream(StringIO(template)) if item.error]
    settings = dotenv_values(stream=StringIO(template))
    assert {"GROQ_API_KEY", "GROQ_MODEL"} <= settings.keys()
    assert settings["GROQ_MODEL"] == "openai/gpt-oss-120b"
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setenv("GROQ_MODEL", settings["GROQ_MODEL"])
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-credential")
    client = llm.get_client()
    assert client.api_key == "test-key-not-a-real-credential"
    assert client.base_url.host == "api.groq.com"
    assert client.base_url.path.rstrip("/") == "/openai/v1"
    assert client.max_retries == 0


def test_paid_model_is_rejected_before_any_client_request(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "gpt-6-sol")
    monkeypatch.setattr(llm, "get_client", lambda: pytest.fail("Invalid models must not construct a client"))
    with pytest.raises(RuntimeError, match="free"):
        llm.chat([{"role": "user", "content": "hello"}])


def test_chat_uses_only_groq_model_and_does_not_retry_quota_errors(monkeypatch):
    from types import SimpleNamespace
    import httpx
    from openai import RateLimitError

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
        raise RateLimitError("Free quota reached", response=response, body=None)

    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(
        llm, "get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    with pytest.raises(RuntimeError, match="quota"):
        llm.chat([{"role": "user", "content": "Explain binary search"}])
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/gpt-oss-120b"
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["extra_body"] == {"include_reasoning": False}


class SmallEmbedder:
    def encode(self, texts, **kwargs):
        return np.array(
            [[1.0, *(float(word in text.lower()) for word in ("binary", "python", "java", "cpp"))] for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def local_index(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    for name in ("_client", "_collection", "_bm25", "_bm25_docs"):
        monkeypatch.setattr(retrieval, name, None)
    monkeypatch.setattr(retrieval, "DB_DIR", tmp_path / "database")
    monkeypatch.setattr(retrieval, "SNAPSHOT", tmp_path / "missing-snapshot.json.gz")
    monkeypatch.setattr(retrieval, "_model", SmallEmbedder())
    return tmp_path


def write_lesson(root, relative, text=None):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        text
        or (
            "# Binary search lesson\n"
            "Binary search repeatedly halves a sorted array. Its invariant keeps the target "
            "inside the remaining interval. Empty arrays return a missing result.\n"
        ),
        encoding="utf-8",
    )
    return path


def test_fresh_deployment_restores_snapshot_without_loading_models(local_index, monkeypatch):
    import gzip
    import json

    seed = local_index / "seed.json.gz"
    snapshot = {
        "embedder": retrieval.MODEL_NAME,
        "ids": ["python/search.py#0"],
        "documents": ["Binary search repeatedly halves the sorted interval."],
        "metadatas": [{"source": "python/search.py", "topic": "searching", "language": "python"}],
        "embeddings": [[1.0, 0.0, 0.0, 0.0, 0.0]],
    }
    with gzip.open(seed, "wt", encoding="utf-8") as file:
        json.dump(snapshot, file)
    monkeypatch.setattr(retrieval, "SNAPSHOT", seed, raising=False)
    monkeypatch.setattr(retrieval, "_get_model", lambda: pytest.fail("A deployment seed must not download or run models"))
    assert retrieval.index_stats()["languages"] == {"python": 1}
    collection = retrieval._get_collection()
    collection.upsert(
        ids=["local"],
        documents=["Local data"],
        metadatas=[{"source": "local", "topic": "local", "language": "cpp"}],
        embeddings=[[0.0, 1.0, 0.0, 0.0, 0.0]],
    )
    monkeypatch.setattr(retrieval, "_collection", None)
    seed.write_bytes(b"broken seed must be ignored for an existing index")
    assert retrieval._get_collection().count() == 2


def test_wrong_snapshot_model_fails_before_import(local_index, monkeypatch):
    import gzip
    import json

    seed = local_index / "seed.json.gz"
    with gzip.open(seed, "wt", encoding="utf-8") as file:
        json.dump({"embedder": "wrong-model"}, file)
    monkeypatch.setattr(retrieval, "SNAPSHOT", seed, raising=False)
    with pytest.raises(RuntimeError, match="embedding model"):
        retrieval._get_collection()
    assert retrieval._collection is None


def test_snapshot_export_roundtrip_and_secret_rejection(local_index, monkeypatch):
    import export_index

    root = local_index / "python"
    write_lesson(root, "search.py")
    count = retrieval.build_index(sources={"python": root})
    destination = local_index / "snapshot.json.gz"
    assert export_index.export_snapshot(destination) == count
    before = destination.read_bytes()
    collection = retrieval._get_collection()
    data = collection.get(include=["embeddings"])
    collection.update(ids=data["ids"][:1], documents=["gsk_" + "x" * 40], embeddings=data["embeddings"][:1].tolist())
    with pytest.raises(RuntimeError, match="credential"):
        export_index.export_snapshot(destination)
    assert destination.read_bytes() == before
    monkeypatch.setattr(retrieval, "DB_DIR", local_index / "restored-database")
    monkeypatch.setattr(retrieval, "SNAPSHOT", destination)
    monkeypatch.setattr(retrieval, "_collection", None)
    monkeypatch.setattr(retrieval, "_client", None)
    assert retrieval.index_stats()["languages"] == {"python": count}


def test_discovers_three_languages_and_ignores_build_artifacts(tmp_path):
    for relative in ("arrays/Search.cpp", "arrays/search.py", "arrays/Search.java"):
        write_lesson(tmp_path, relative)
    for relative in (
        ".git/secret.py",
        ".dsa_compile_temp/temp.py",
        "out/Generated.java",
        "__pycache__/cached.py",
        "verify_collection.py",
    ):
        write_lesson(tmp_path, relative)
    chunks = chunking.chunk_repo(tmp_path)
    assert {item["language"] for item in chunks} == {"cpp", "python", "java"}
    assert {item["source"] for item in chunks} == {"arrays/Search.cpp", "arrays/search.py", "arrays/Search.java"}


def test_python_chunks_preserve_decorators_and_indentation(tmp_path):
    source = '"""Binary search and supporting explanations for a sorted sequence of values."""\n\n'
    source += '@staticmethod\ndef search(values, target):\n    """Find the first matching value using a sorted sequence."""\n    return -1\n'
    source += '\ndef bounds(values, target):\n    """Calculate the lower and upper insertion positions for a sorted sequence."""\n    return (0, len(values))\n'
    path = write_lesson(tmp_path, "search.py", source)
    chunks = chunking.chunk_file(path, tmp_path)
    assert any("@staticmethod\ndef search" in c["text"] for c in chunks)
    assert any("    return (0, len(values))" in c["text"] for c in chunks)


def test_multiple_roots_get_unique_sources_and_language_filter(local_index):
    sources = {}
    for language in ("cpp", "python", "java"):
        root = local_index / language
        write_lesson(root, "arrays/README.md")
        write_lesson(root, {"cpp": "Search.cpp", "python": "search.py", "java": "Search.java"}[language])
        sources[language] = root
    count = retrieval.build_index(sources=sources)
    assert count >= 6
    collection = retrieval._get_collection()
    ids = collection.get(include=[])["ids"]
    assert len(ids) == len(set(ids)) == count
    assert set(retrieval.index_stats()["languages"]) == {"cpp", "python", "java"}
    for language in sources:
        hits = retrieval.hybrid_search("binary search", language=language, rerank=False)
        assert hits
        assert all(h["language"] == language and h["source"].startswith(language + "/") for h in hits)
    assert retrieval.hybrid_search("binary search", k=20, candidates=20, rerank=False)


def test_reindex_removes_old_chunks_and_keeps_live_collection(local_index):
    root = local_index / "python"
    path = write_lesson(root, "search.py", "# Binary search explanation\n" * 180)
    other = write_lesson(root, "removed.py")
    first_count = retrieval.build_index(sources={"python": root})
    collection = retrieval._get_collection()
    path.write_text("# Binary search explanation\n" * 6, encoding="utf-8")
    other.unlink()
    second_count = retrieval.build_index(sources={"python": root}, force=True)
    assert second_count < first_count
    assert retrieval._get_collection() is collection
    assert collection.count() == second_count
    assert all("removed.py" not in value for value in collection.get(include=[])["ids"])
    assert retrieval.hybrid_search("binary search", language="python", rerank=False)


def test_invalid_rebuild_and_embedding_failure_preserve_existing_index(local_index, monkeypatch):
    root = local_index / "python"
    write_lesson(root, "search.py")
    retrieval.build_index(sources={"python": root})
    collection = retrieval._get_collection()
    before = collection.get(include=["documents", "metadatas"])
    with pytest.raises(RuntimeError):
        retrieval.build_index(sources={"java": local_index / "missing"}, force=True)

    class BrokenEmbedder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(retrieval, "_model", BrokenEmbedder())
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        retrieval.build_index(sources={"python": root}, force=True)
    after = collection.get(include=["documents", "metadatas"])
    assert after == before


def test_search_validation(local_index):
    with pytest.raises(ValueError):
        retrieval.hybrid_search("binary search", language="ruby")
    with pytest.raises(ValueError):
        retrieval.hybrid_search("binary search", k=0)


def test_fusion_keeps_distinct_chunks_with_the_same_long_header():
    header = "Source: java/src/dynamic_programming/Subsequences.java " * 3
    first = {"source": "java/Subsequences.java", "text": header + "LCS implementation", "distance": 0.1}
    second = {"source": "java/Subsequences.java", "text": header + "LIS implementation", "distance": 0.2}
    hits = retrieval._rrf_fuse([first, second], [first.copy(), second.copy()])
    assert len(hits) == 2
    assert {h["distance"] for h in hits} == {0.1, 0.2}
    hits[0]["relevance"] = 10
    assert "relevance" not in first and "_key" not in first


def test_answer_uses_requested_language_and_correct_fences(monkeypatch):
    messages = []
    monkeypatch.setattr(llm, "chat", lambda request: messages.extend(request) or "Use [java/arrays/Search.java].")
    contexts = [{"source": "java/arrays/Search.java", "text": "Binary search implementation", "language": "java"}]
    result = llm.answer("Explain binary search", contexts, language="java")
    assert "Java" in messages[0]["content"]
    assert "```java" in messages[0]["content"]
    assert "[Source: java/arrays/Search.java]" in messages[-1]["content"]
    assert result["sources"] == ["java/arrays/Search.java"]


def test_empty_context_does_not_call_model(monkeypatch):
    def should_not_call(*args, **kwargs):
        pytest.fail("An empty retrieval should not call the model")

    monkeypatch.setattr(llm, "chat", should_not_call)
    result = llm.answer("binary search", [], language="python")
    assert result["answer"] == "This isn't in my notes yet"
    assert result["sources"] == []


def test_ingest_cli_passes_explicit_language_roots(tmp_path, monkeypatch):
    import ingest

    calls = []
    monkeypatch.setattr("sys.argv", ["ingest.py", "--python-dir", str(tmp_path), "--java-dir", str(tmp_path / "src")])
    monkeypatch.setattr(ingest, "build_index", lambda **kwargs: calls.append(kwargs) or 2)
    monkeypatch.setattr(ingest, "index_stats", lambda: {"chunks": 2})
    ingest.main()
    assert calls[0]["sources"] == {"python": tmp_path, "java": tmp_path / "src"}


def test_streamlit_reuses_context_and_keeps_language_conversations(monkeypatch):
    from streamlit.testing.v1 import AppTest

    requests = []
    monkeypatch.setattr(
        retrieval,
        "index_stats",
        lambda: {
            "chunks": 3,
            "languages": {"cpp": 1, "python": 1, "java": 1},
            "embedder": "test",
            "reranker": "test",
        },
    )

    def retrieve(question, **kwargs):
        language = kwargs.get("language") or "cpp"
        requests.append(language)
        return [
            {
                "text": "Binary search explanation",
                "source": f"{language}/search.py",
                "topic": "searching",
                "language": language,
            }
        ]

    monkeypatch.setattr(retrieval, "hybrid_search", retrieve)
    monkeypatch.setattr(
        llm,
        "answer",
        lambda question, hits, **kwargs: {"answer": "Halve the sorted interval.", "sources": [h["source"] for h in hits]},
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.selectbox[0].select("Python").run()
    app.chat_input[0].set_value("Explain binary search").run()
    assert not app.exception
    assert requests == ["python"]
    app.selectbox[0].select("Java").run()
    assert len(app.chat_message) == 0
    app.selectbox[0].select("Python").run()
    assert len(app.chat_message) == 2
    assert requests == ["python"]
    app.button(key="clear_conversation").click().run()
    assert not app.exception
    assert not app.chat_message
    assert not app.session_state["conversations"]["python"]["history"]
    app.button(key="suggest_blue").click().run()
    assert not app.exception
    assert requests == ["python", "python"]
    assert len(app.chat_message) == 2
    assert app.session_state["conversations"]["python"]["question_count"] == 1
