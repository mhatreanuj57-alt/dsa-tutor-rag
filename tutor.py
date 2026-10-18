import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from src.llm import answer
from src.retrieval import hybrid_search, index_stats


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="DSA Tutor - ask about your notes")
    parser.add_argument("question", nargs="*", help="Question(s) to ask")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve (default: 5)")
    parser.add_argument("--language", choices=("cpp", "python", "java"), help="Restrict answers to one language")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stats:
        print(index_stats())
        return

    if args.question:
        questions = [" ".join(args.question)]
    else:
        questions = None
        print("DSA Tutor (hybrid + reranker) - ask about your notes (blank to quit)\n")

    while True:
        q = questions[0] if questions else input("you > ").strip()
        if not q:
            break

        print(f"\nretrieving (hybrid + rerank, top_k={args.top_k})...")
        hits = hybrid_search(q, k=args.top_k, language=args.language)
        print("top sources:")
        for h in hits[:3]:
            score = h.get("relevance", h.get("distance"))
            print(f"  - {h['source']} ({score})")

        result = answer(q, hits, language=args.language)
        print(f"\n{result['answer']}\n")
        print("cited:", ", ".join(dict.fromkeys(result["sources"])), "\n")

        if questions:
            break


if __name__ == "__main__":
    main()
