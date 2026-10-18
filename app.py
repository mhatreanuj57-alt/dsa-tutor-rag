import base64
import json
import sys
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.chunking import FILE_LANGUAGES
from src.llm import answer
from src.retrieval import hybrid_search, index_stats

st.set_page_config(page_title="DSA tutor | Your study workspace", page_icon=":material/school:", layout="wide")
font = base64.b64encode((ROOT / "assets/material-icons-outlined.otf").read_bytes()).decode("ascii")
st.html(
    "<style>@font-face{font-family:'Tutor Material';src:url(data:font/otf;base64,"
    + font
    + ") format('opentype');font-weight:400;font-style:normal;font-display:block;}"
    + (ROOT / "assets/ui.css").read_text(encoding="utf-8")
    + "</style>"
)

with st.sidebar:
    st.html(
        '<div class="brand"><span class="brand-symbol mi" aria-hidden="true">school</span>'
        '<span>DSA <strong>tutor</strong><span class="brand-caption">A space for your curiosity</span></span></div>'
    )
    st.html('<div class="sidebar-label">YOUR WORKSPACE</div>')
    languages = {"All languages": None, "C++": "cpp", "Python": "python", "Java": "java"}
    selected = st.selectbox("Study language", list(languages), key="study_language")
    language = languages[selected]

try:
    stats = index_stats()
except RuntimeError as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

available = stats["chunks"] if language is None else stats["languages"].get(language, 0)
if "conversations" not in st.session_state:
    st.session_state.conversations = {
        "all": {
            "messages": st.session_state.get("messages", []),
            "history": st.session_state.get("history", []),
            "question_count": st.session_state.get("question_count", 0),
        }
    }
conversation = st.session_state.conversations.setdefault(
    language or "all", {"messages": [], "history": [], "question_count": 0}
)

with st.sidebar:
    st.html('<div class="sidebar-label library-label">YOUR LIBRARY</div>')
    for label, code, color in (("C++", "cpp", "blue"), ("Python", "python", "yellow"), ("Java", "java", "green")):
        count = stats["languages"].get(code, 0)
        st.html(
            f'<div class="library-row"><span class="language-icon {color}" aria-hidden="true">'
            f'<span class="mi">code</span></span><span>{label}</span><span class="library-count">{count:,}</span></div>'
        )
    st.caption("Indexed passages from your own notes")
    actions = st.container(key="session_actions")
    st.html(
        '<div class="sidebar-note"><span class="mi" aria-hidden="true">lightbulb</span>'
        "<strong>Learn with evidence.</strong><p>Every answer starts with your notes. Open a source to see the original code.</p></div>"
        '<div class="sidebar-signature"><span class="color-dot blue-dot"></span>'
        '<span class="color-dot red-dot"></span><span class="color-dot yellow-dot"></span>'
        '<span class="color-dot green-dot"></span><span>Keep asking. Keep learning.</span></div>'
    )

st.html(
    '<div class="workspace-bar"><span><span class="mi" aria-hidden="true">chat_bubble_outline</span>'
    f'Study workspace <span class="breadcrumb">/ {escape(selected)}</span></span>'
    f'<span class="library-status"><span class="status-dot"></span>{available:,} passages ready</span></div>'
)

if not available:
    st.info(
        "No notes are indexed for this language. Run ingest.py with your collection folder to get started.",
        icon=":material/menu_book:",
    )

question = st.chat_input("Ask a question about your notes…", disabled=not available, submit_mode="disable")
workspace = st.container(key="study_layout")
chat_col, side_col = workspace.columns([2.6, 1], gap="large")
suggested_question = None

with chat_col:
    welcome = st.empty()
    if not conversation["messages"] and not question:
        with welcome.container():
            st.html(
                '<div class="welcome"><div class="welcome-eyebrow"><span class="color-dot blue-dot"></span>'
                '<span class="color-dot red-dot"></span><span class="color-dot yellow-dot"></span>'
                '<span class="color-dot green-dot"></span> A LITTLE CURIOSITY GOES A LONG WAY</div>'
                "<h1>Big ideas.<br><span>Clear explanations.</span></h1>"
                "<p>Your code. Your notes. A little help connecting the dots.<br>"
                "Explore data structures and algorithms, one question at a time.</p></div>"
            )
            st.html('<div class="suggestion-label">A good place to start</div>')
            suggestions = (
                ("Trace binary search", "Walk me through binary search in my notes with an example.", "search", "blue"),
                ("Understand recursion", "Explain recursion using an implementation in my notes.", "hub", "red"),
                (
                    "Explore a sorting algorithm",
                    "Explain a sorting algorithm from my notes and its time complexity.",
                    "code",
                    "yellow",
                ),
                ("Make sense of graphs", "Explain a graph traversal from my notes, step by step.", "account_tree", "green"),
            )
            for row in (suggestions[:2], suggestions[2:]):
                columns = st.columns(2, gap="small")
                for column, (label, prompt, icon, color) in zip(columns, row):
                    with column:
                        if st.button(
                            label, icon=f":material/{icon}:", key=f"suggest_{color}", width="stretch", disabled=not available
                        ):
                            suggested_question = prompt
            st.html('<p class="welcome-footnote">Choose a language in the sidebar, or explore them together.</p>')

    for message in conversation["messages"]:
        avatar = ":material/school:" if message["role"] == "assistant" else ":material/account_circle:"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + " · ".join(message["sources"]))

    if question := question or suggested_question:
        welcome.empty()
        conversation["question_count"] += 1
        conversation["messages"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar=":material/account_circle:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/school:"):
            with st.spinner("Finding the right notes and connecting the dots…"):
                try:
                    hits = hybrid_search(question, k=5, language=language)
                    result = answer(question, hits, history=conversation["history"], language=language)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        st.caption("Sources: " + " · ".join(result["sources"]))
                    conversation["messages"].append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"], "contexts": hits}
                    )
                    conversation["history"].extend(
                        [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]
                    )
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))

with side_col:
    with st.container(key="evidence", border=True):
        st.html(
            '<div class="evidence-heading"><span class="mi" aria-hidden="true">menu_book</span><h2>Behind the answer</h2></div>'
        )
        last = conversation["messages"][-1] if conversation["messages"] else {}
        hits = last.get("contexts", [])
        if hits:
            st.caption("The exact passages used for this answer")
            for i, hit in enumerate(hits):
                with st.expander(f"{i + 1}. {Path(hit['source']).name}", icon=":material/description:"):
                    st.caption(hit["source"])
                    syntax = FILE_LANGUAGES.get(Path(hit["source"]).suffix.lower(), "text")
                    st.code(hit["text"], language=syntax)
        else:
            st.html(
                '<div class="evidence-empty"><span class="mi" aria-hidden="true">bookmarks</span>'
                "<h3>Your notes take the lead.</h3><p>Ask a question and the supporting code will appear right here.</p></div>"
                '<div class="evidence-step"><span class="mi blue-text" aria-hidden="true">search</span>'
                "<div><strong>Find the relevant notes</strong><span>Across your selected language</span></div></div>"
                '<div class="evidence-step"><span class="mi yellow-text" aria-hidden="true">code</span>'
                "<div><strong>Connect it to your code</strong><span>Real implementations, in context</span></div></div>"
                '<div class="evidence-step"><span class="mi green-text" aria-hidden="true">school</span>'
                "<div><strong>Understand the why</strong><span>Clear explanations with citations</span></div></div>"
            )
        st.html(
            '<div class="evidence-bottom"><span class="mi" aria-hidden="true">description</span> Sources you can inspect</div>'
        )
    count = conversation["question_count"]
    st.caption(f"{count} {'question' if count == 1 else 'questions'} in this conversation")

with actions:
    st.html('<div class="sidebar-label">THIS CONVERSATION</div>')
    st.download_button(
        "Download chat",
        data=json.dumps(conversation["messages"], indent=2, ensure_ascii=False),
        file_name=f"dsa_tutor_{language or 'all'}_chat.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
        disabled=not conversation["messages"],
    )
    if st.button(
        "Clear conversation",
        icon=":material/delete_outline:",
        key="clear_conversation",
        width="stretch",
        disabled=not conversation["messages"],
    ):
        conversation["messages"].clear()
        conversation["history"].clear()
        conversation["question_count"] = 0
        st.rerun()

st.html(
    '<p class="answer-note">A study companion, grounded in your notes. Check the sources and keep thinking critically.</p>'
)
