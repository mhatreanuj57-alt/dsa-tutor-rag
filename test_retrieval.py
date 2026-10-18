import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import search

for q in [
    "how does Kruskal minimum spanning tree work",
    "what is binary search tree insertion",
    "explain dynamic programming memoization",
]:
    print(f"Q: {q}")
    for h in search(q, k=3):
        print(f"  [{h.get('relevance', h.get('distance', '?'))}] {h['source']}  ->  {h['text'][:100].replace(chr(10), ' ')}")
    print()
