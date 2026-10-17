import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parent

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.llm import answer, get_client, get_model
from src.retrieval import hybrid_search

print(f"model: {get_model()}")
print("testing connection...")
client = get_client()
probe = client.chat.completions.create(
    model=get_model(),
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=10,
)
print(f"connection: {probe.choices[0].message.content}")
print()

question = "How does Kruskal's minimum spanning tree algorithm work?"
print(f"Q: {question}\n")
hits = hybrid_search(question, k=5)
print("retrieved:", [h["source"] for h in hits], "\n")
result = answer(question, hits)
print("=== ANSWER ===")
print(result["answer"])
print()
print("cited sources:", list(dict.fromkeys(result["sources"])))
