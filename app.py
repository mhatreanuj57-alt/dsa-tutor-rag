import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parent

import json
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.llm import answer
from src.retrieval import hybrid_search, index_stats

st.set_page_config(page_title="DSA Tutor - RAG", page_icon="🤖", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.hero { background: linear-gradient(135deg, #1E1B4B 0%, #4F46E5 100%);
        border-radius: 16px; padding: 26px 34px; margin-bottom: 20px; }
.hero h1 { color: #fff; font-size: 1.8rem; font-weight: 800; margin: 0; }
.hero p { color: #C7D2FE; font-size: 0.92rem; margin: 6px 0 0 0; }
.src-chip { background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 999px;
            padding: 4px 14px; font-size: 0.8rem; color: #3730A3; font-weight: 600;
            display: inline-block; margin: 3px 4px 3px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading index + models...")
def load_stats():
    return index_stats()


try:
    stats = load_stats()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.markdown(
    '<div class="hero"><h1>🤖 DSA Tutor — RAG over my own notes</h1>'
    f'<p>{stats["chunks"]} chunks · hybrid search (vector + BM25 + RRF) · '
    f'cross-encoder reranked · grounded answers with citations</p></div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

chat_col, side_col = st.columns([2.6, 1], gap="large")

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                chips = "".join(
                    f'<span class="src-chip">📄 {s}</span>' for s in msg["sources"][:5]
                )
                st.markdown(chips, unsafe_allow_html=True)

    if question := st.chat_input("Ask anything from your DSA notes... e.g. how does Kruskal work?"):
        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("hybrid retrieval → rerank → generate..."):
                try:
                    hits = hybrid_search(question, k=5)
                    result = answer(question, hits, history=st.session_state.history)
                    st.markdown(result["answer"])
                    chips = "".join(
                        f'<span class="src-chip">📄 {s}</span>' for s in result["sources"][:5]
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                    st.session_state.history.append({"role": "user", "content": question})
                    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
                except RuntimeError as e:
                    st.error(str(e))

with side_col:
    st.subheader("📚 Retrieved context")
    st.caption("Top chunks behind the last answer")
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        try:
            hits = hybrid_search(st.session_state.messages[-2]["content"], k=5)
            for i, h in enumerate(hits):
                rel = h.get("relevance")
                label = f"{'⭐' if i == 0 else '📄'} {h['source']}"
                with st.expander(label):
                    if rel is not None:
                        st.caption(f"reranker score: {rel}")
                    st.code(h["text"][:600], language="cpp")
        except Exception:
            st.caption("Ask a question to see retrieval")
    else:
        st.caption("Ask a question to see retrieval")
    
    st.divider()
    st.subheader("📊 Session Stats")
    st.metric("Questions Asked", st.session_state.question_count)
    if st.session_state.messages:
        st.metric("Messages", len(st.session_state.messages))
    
    st.divider()
    st.subheader("💾 Export")
    if st.session_state.messages:
        export_data = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Chat History",
            data=export_data,
            file_name="dsa_tutor_chat.json",
            mime="application/json",
        )
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.question_count = 0
            st.rerun()

st.divider()
c1, c2, c3 = st.columns(3)
c1.caption(f"🧠 chunks: {stats['chunks']}")
c2.caption(f"🔎 embedder: {stats['embedder']}")
c3.caption(f"⚖️ reranker: {stats['reranker']}")
