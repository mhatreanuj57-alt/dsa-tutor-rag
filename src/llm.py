import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """You are a DSA Tutor - a patient data structures & algorithms teacher for a \
second-year BTech student. You answer ONLY using the provided context, which consists of the \
student's own C++ implementations and notes.

Rules:
- Explain step by step, simply, like a friendly senior would.
- Always cite sources as [filename] after every claim taken from context.
- If the context contains relevant C++ code, show it in a ```cpp block.
- If the answer is not in the context, say exactly: "This isn't in my notes yet" and suggest \
what topic to add. Never invent content.
- Keep answers under 300 words unless code is needed.
"""

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("ZEN_API_KEY")
        if not key:
            raise RuntimeError("Set ZEN_API_KEY in .env first")
        base = os.environ.get("ZEN_BASE_URL", "https://opencode.ai/zen/v1")
        _client = OpenAI(api_key=key, base_url=base)
    return _client


def get_model():
    return os.environ.get("ZEN_MODEL", "nemotron-3-ultra-free")


def chat(messages: list, temperature: float = 0.3, retries: int = 2) -> str:
    client = get_client()
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=get_model(),
                messages=messages,
                temperature=temperature,
                max_tokens=1200,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            last_error = RuntimeError("empty response from model")
        except Exception as e:
            last_error = e
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}")


def answer(question: str, contexts: list, history: list = None) -> dict:
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in contexts
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": f"Context from student's notes:\n\n{context_block}\n\n"
                   f"Question: {question}\n\n"
                   "Answer with citations [filename] for every claim from the context.",
    })
    return {"answer": chat(messages), "sources": [c["source"] for c in contexts]}
