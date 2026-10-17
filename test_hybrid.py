import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import hybrid_search, index_stats

print("stats:", index_stats())
print()

queries = [
    "how does Kruskal minimum spanning tree work",
    "binary search tree insertion",
    "KMP string matching algorithm",
]
for q in queries:
    t0 = time.time()
    hits = hybrid_search(q, k=5)
    dt = time.time() - t0
    print(f"Q: {q}  ({dt:.1f}s)")
    for i, h in enumerate(hits):
        star = "*" if i == 0 else " "
        print(f"  {star} [{h.get('relevance', '?'):>7}] {h['source']}")
    print()
