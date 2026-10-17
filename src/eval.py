import json

from src.llm import chat, get_model
from src.retrieval import hybrid_search

EVAL_QUESTIONS = [
    "How does Kruskal's minimum spanning tree algorithm work?",
    "What is binary search tree insertion?",
    "Explain dynamic programming memoization with an example",
    "How does Dijkstra's algorithm find shortest paths?",
    "What is a hash map and how does it handle collisions?",
    "Explain the difference between BFS and DFS",
    "How does quicksort partitioning work?",
    "What is a segment tree used for?",
]


def judge(question: str, answer: str, contexts: list) -> dict:
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text'][:800]}" for c in contexts
    )
    prompt = f"""Rate this RAG answer on two scales from 0 to 1.

Question: {question}

Retrieved context:
{context_block}

Answer:
{answer}

Respond with ONLY a JSON object:
{{"faithfulness": <0-1, is the answer grounded ONLY in the context, no invented facts>,
  "relevance": <0-1, does the answer address the question>,
  "cited": <true/false, does the answer cite [filenames]>}}"""
    raw = chat([{"role": "user", "content": prompt}], temperature=0.0)
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, TypeError):
        return {"faithfulness": None, "relevance": None, "cited": None, "raw": raw}


def run_eval(k: int = 5):
    from src.llm import answer
    results = []
    for q in EVAL_QUESTIONS:
        try:
            hits = hybrid_search(q, k=k)
            if not hits:
                results.append({"question": q, "error": "no retrieval"})
                continue
            result = answer(q, hits)
            verdict = judge(q, result["answer"], hits)
            results.append({
                "question": q,
                "faithfulness": verdict.get("faithfulness"),
                "relevance": verdict.get("relevance"),
                "cited": verdict.get("cited"),
                "top_source": hits[0]["source"],
            })
            print(f"Q: {q[:50]}... -> faith={verdict.get('faithfulness')} "
                  f"rel={verdict.get('relevance')} src={hits[0]['source']}")
        except Exception as e:
            results.append({"question": q, "error": str(e)[:100]})
            print(f"Q: {q[:50]}... -> ERROR {str(e)[:80]}")
    return results


def summarize(results: list) -> dict:
    faith = [r["faithfulness"] for r in results if isinstance(r.get("faithfulness"), (int, float))]
    rel = [r["relevance"] for r in results if isinstance(r.get("relevance"), (int, float))]
    cited = [r["cited"] for r in results if isinstance(r.get("cited"), bool)]
    return {
        "n": len(results),
        "avg_faithfulness": round(sum(faith) / len(faith), 3) if faith else None,
        "avg_relevance": round(sum(rel) / len(rel), 3) if rel else None,
        "cited_pct": round(100 * sum(cited) / len(cited), 1) if cited else None,
    }


if __name__ == "__main__":
    results = run_eval()
    print("\n=== SUMMARY ===")
    print(json.dumps(summarize(results), indent=2))
