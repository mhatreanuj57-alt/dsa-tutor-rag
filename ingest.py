import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import build_index, index_stats


def main():
    parser = argparse.ArgumentParser(description="Index C++, Python, and Java DSA notes")
    parser.add_argument("--notes-dir", type=Path, help="One mixed notes folder (default: data/notes)")
    for language in ("cpp", "python", "java"):
        parser.add_argument(f"--{language}-dir", type=Path, help=f"Directory containing {language} lessons")
    parser.add_argument("--force", action="store_true", help="Refresh the complete index (legacy alias)")
    args = parser.parse_args()
    sources = {
        language: getattr(args, f"{language}_dir")
        for language in ("cpp", "python", "java")
        if getattr(args, f"{language}_dir") is not None
    }
    if args.notes_dir and sources:
        parser.error("Use --notes-dir or language directories, not both")
    n = build_index(notes_dir=args.notes_dir, sources=sources or None, force=args.force)
    print(f"Indexed {n} chunks")
    print(index_stats())


if __name__ == "__main__":
    main()
