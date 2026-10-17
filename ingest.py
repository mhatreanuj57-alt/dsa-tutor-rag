import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import build_index, index_stats

if __name__ == "__main__":
    force = "--force" in sys.argv
    n = build_index(force=force)
    print(f"Indexed {n} chunks")
    print(index_stats())
