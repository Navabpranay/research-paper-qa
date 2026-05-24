import streamlit as st
from dotenv import load_dotenv
from src.fetcher import fetch_and_download_papers
from src.processor import process_papers, process_uploaded_pdf
from src.embedder import add_papers_to_store, load_vector_store
from src.generator import answer_question
from src.evaluator import log_interaction
from src.summarizer import generate_all_summaries, load_paper_summaries
from src.memory import (
    initialize_chat_history,
    format_history_for_display,
    answer_with_memory,
    clear_history
)
from src.exporter import export_as_text, export_as_pdf
import os
import tempfile
from datetime import datetime

load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Paper Q&A",
    page_icon="📚",
    layout="wide"
)

# ─────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = initialize_chat_history()

if "loaded_papers" not in st.session_state:
    st.session_state["loaded_papers"] = []

if "paper_summaries" not in st.session_state:
    st.session_state["paper_summaries"] = load_paper_summaries()

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📚 AI Research Paper Summarizer & Q&A")
st.markdown(
    "*Search ArXiv papers or upload your own — "
    "ask questions with full conversation memory*"
)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:

    # Section 1 — ArXiv Search
    st.header("🔍 Search ArXiv Papers")

    research_topic = st.text_input(
        "Research Topic",
        placeholder="e.g., LLM hallucination reduction"
    )
    max_papers = st.slider("Number of Papers", 2, 10, 5)

    if st.button("🔍 Fetch & Index Papers", type="primary"):
        if not research_topic.strip():
            st.error("Please enter a research topic.")
        else:
            with st.spinner("Searching ArXiv..."):
                papers = fetch_and_download_papers(
                    research_topic, max_papers
                )

            with st.spinner("Processing and indexing..."):
                chunks = process_papers(papers)
                vector_store = add_papers_to_store(chunks)
                st.session_state["vector_store"] = vector_store
                st.session_state["loaded_papers"] = papers

            with st.spinner(
                "Generating paper summaries... (this takes ~30 seconds)"
            ):
                summaries = generate_all_summaries(
                    papers, vector_store
                )
                st.session_state["paper_summaries"] = summaries

            # Reset chat when new papers loaded
            st.session_state["chat_history"] = initialize_chat_history()
            st.session_state["last_result"] = None

            st.success(f"✅ Indexed {len(papers)} papers!")

    st.divider()

    # Section 2 — Upload PDF
    st.header("📤 Upload Your Own Paper")
    st.caption("Upload PDFs not available on ArXiv")

    uploaded_files = st.file_uploader(
        "Upload PDF Research Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Papers", type="primary"):
            all_chunks = []
            uploaded_paper_metas = []
            progress = st.progress(0)
            status = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status.text(f"Processing: {uploaded_file.name}")

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                    dir="data/papers"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                paper_metadata = {
                    "title": uploaded_file.name.replace(".pdf", ""),
                    "authors": ["Uploaded by User"],
                    "arxiv_id": "",
                    "published": "User Upload",
                    "local_path": tmp_path
                }

                chunks = process_uploaded_pdf(tmp_path, paper_metadata)
                all_chunks.extend(chunks)
                uploaded_paper_metas.append(paper_metadata)
                progress.progress((i + 1) / len(uploaded_files))

            status.text("Indexing...")
            with st.spinner("Adding to knowledge base..."):
                vector_store = add_papers_to_store(all_chunks)
                st.session_state["vector_store"] = vector_store
                st.session_state["loaded_papers"] += uploaded_paper_metas

            with st.spinner("Generating summaries..."):
                summaries = generate_all_summaries(
                    uploaded_paper_metas, vector_store
                )
                st.session_state["paper_summaries"].update(summaries)

            st.session_state["chat_history"] = initialize_chat_history()
            progress.empty()
            status.empty()
            st.success(
                f"✅ Indexed {len(uploaded_files)} uploaded paper(s)!"
            )

    st.divider()

    # Section 3 — Load Existing
    st.header("📂 Load Existing Index")

    if st.button("📂 Load Existing Index"):
        try:
            vector_store = load_vector_store()
            st.session_state["vector_store"] = vector_store
            st.session_state["paper_summaries"] = load_paper_summaries()
            st.session_state["chat_history"] = initialize_chat_history()
            st.success("✅ Loaded existing index!")
        except FileNotFoundError:
            st.error("No existing index found.")

    st.divider()

    # Section 4 — Chat Controls
    if "vector_store" in st.session_state:
        st.header("💬 Chat Controls")

        history_count = len(st.session_state["chat_history"]) // 2
        st.caption(f"Messages in memory: {history_count}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state["chat_history"] = clear_history()
            st.session_state["last_result"] = None
            st.success("Chat history cleared!")
            st.rerun()

# ─────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────
if "vector_store" not in st.session_state:

    st.info(
        "👈 Use the sidebar to get started:\n\n"
        "**Option 1:** Search ArXiv papers\n\n"
        "**Option 2:** Upload your own PDFs\n\n"
        "**Option 3:** Load a previous index"
    )

else:

    # ── TABS ──────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "💬 Chat with Papers",
        "📋 Paper Summaries",
        "📊 Session Stats"
    ])

    # ──────────────────────────────────────
    # TAB 1 — CHAT
    # ──────────────────────────────────────
    with tab1:

        st.header("💬 Chat with Your Papers")
        st.caption(
            "Ask follow-up questions — I remember our full conversation."
        )

        # Quick question buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧠 Hallucination methods?"):
                st.session_state["quick_q"] = (
                    "What methods do these papers propose "
                    "to reduce hallucinations in LLMs?"
                )
        with col2:
            if st.button("📊 Evaluation metrics?"):
                st.session_state["quick_q"] = (
                    "What evaluation metrics are used "
                    "to measure model performance?"
                )
        with col3:
            if st.button("🔍 Key findings?"):
                st.session_state["quick_q"] = (
                    "What are the key findings across these papers?"
                )

        st.divider()

        # Display full chat history
        chat_display = format_history_for_display(
            st.session_state["chat_history"]
        )

        if chat_display:
            st.subheader("🗨️ Conversation History")
            for msg in chat_display:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
            st.divider()

        # Question input
        question = st.text_area(
            "Your Question",
            value=st.session_state.pop("quick_q", ""),
            placeholder=(
                "Ask anything... I remember previous questions in this session."
            ),
            height=100,
            key="question_input"
        )

        col_ask, col_space = st.columns([1, 4])
        with col_ask:
            ask_clicked = st.button(
                "🔮 Ask", type="primary", use_container_width=True
            )

        if ask_clicked and question.strip():

            with st.spinner("Thinking..."):
                answer, updated_history, chunks = answer_with_memory(
                    question=question,
                    vector_store=st.session_state["vector_store"],
                    chat_history=st.session_state["chat_history"]
                )
                st.session_state["chat_history"] = updated_history

                # Save result for export
                result = {
                    "question": question,
                    "answer": answer,
                    "sources": [
                        {
                            "title": c["metadata"].get("title", ""),
                            "authors": c["metadata"].get("authors", ""),
                            "arxiv_id": c["metadata"].get("arxiv_id", ""),
                            "similarity_score": c["similarity_score"]
                        }
                        for c in chunks
                    ],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state["last_result"] = result
                log_interaction(result)

            st.rerun()

        elif ask_clicked and not question.strip():
            st.warning("Please enter a question.")

        # ── EXPORT SECTION ────────────────
        if st.session_state["last_result"]:
            result = st.session_state["last_result"]

            st.divider()
            st.subheader("📥 Export Last Answer")

            col_txt, col_pdf, col_space = st.columns([1, 1, 3])

            # Text export
            with col_txt:
                text_content = export_as_text(
                    question=result["question"],
                    answer=result["answer"],
                    sources=result["sources"]
                )
                st.download_button(
                    label="⬇️ Download .txt",
                    data=text_content,
                    file_name=f"qa_export_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            # PDF export
            with col_pdf:
                try:
                    pdf_bytes = export_as_pdf(
                        question=result["question"],
                        answer=result["answer"],
                        sources=result["sources"]
                    )
                    st.download_button(
                        label="⬇️ Download .pdf",
                        data=pdf_bytes,
                        file_name=f"qa_export_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF export error: {e}")

            # Show sources
            st.subheader("📎 Sources for Last Answer")
            for i, source in enumerate(result["sources"], 1):
                arxiv_id = source.get("arxiv_id", "")
                title = source.get("title", "Unknown")
                label = (
                    f"📄 Source {i}: {title[:50]}"
                    if arxiv_id
                    else f"📤 Source {i} (Uploaded): {title[:50]}"
                )
                with st.expander(label):
                    st.write(f"**Authors:** {source.get('authors', 'N/A')}")
                    st.write(
                        f"**Relevance Score:** "
                        f"{source.get('similarity_score', 'N/A')}"
                    )
                    if arxiv_id:
                        st.markdown(
                            f"[🔗 View on ArXiv]"
                            f"(https://arxiv.org/abs/{arxiv_id})"
                        )

    # ──────────────────────────────────────
    # TAB 2 — PAPER SUMMARIES
    # ──────────────────────────────────────
    with tab2:
        st.header("📋 Paper Summaries")
        st.caption(
            "Auto-generated summaries of all loaded papers."
        )

        summaries = st.session_state.get("paper_summaries", {})

        if not summaries:
            st.info(
                "No summaries yet. "
                "Fetch or upload papers to generate summaries."
            )
        else:
            for paper_id, data in summaries.items():
                title = data.get("title", "Unknown Paper")
                authors = data.get("authors", [])
                published = data.get("published", "")
                arxiv_id = data.get("arxiv_id", "")
                summary = data.get("summary", "No summary available.")

                with st.expander(f"📄 {title[:70]}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if isinstance(authors, list):
                            st.caption(
                                f"👤 {', '.join(authors[:3])}"
                            )
                        else:
                            st.caption(f"👤 {authors}")
                    with col_b:
                        st.caption(f"📅 {published}")

                    if arxiv_id:
                        st.markdown(
                            f"[🔗 View on ArXiv]"
                            f"(https://arxiv.org/abs/{arxiv_id})"
                        )

                    st.markdown("**Summary:**")
                    st.write(summary)

                    # Ask question about this specific paper
                    if st.button(
                        f"💬 Ask about this paper",
                        key=f"ask_{paper_id}"
                    ):
                        st.session_state["quick_q"] = (
                            f"Tell me more about the paper titled '{title}' "
                            f"— what is its main contribution?"
                        )
                        st.rerun()

    # ──────────────────────────────────────
    # TAB 3 — SESSION STATS
    # ──────────────────────────────────────
    with tab3:
        st.header("📊 Session Statistics")

        chat_history = st.session_state.get("chat_history", [])
        summaries = st.session_state.get("paper_summaries", {})
        last_result = st.session_state.get("last_result")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Papers Loaded",
                len(summaries)
            )
        with col2:
            st.metric(
                "💬 Questions Asked",
                len(chat_history) // 2
            )
        with col3:
            st.metric(
                "🧠 Messages in Memory",
                len(chat_history)
            )
        with col4:
            if last_result:
                avg_score = sum(
                    s.get("similarity_score", 0)
                    for s in last_result["sources"]
                ) / len(last_result["sources"]) \
                    if last_result["sources"] else 0
                st.metric(
                    "🎯 Last Avg Score",
                    f"{avg_score:.3f}"
                )
            else:
                st.metric("🎯 Last Avg Score", "N/A")

        st.divider()

        if chat_history:
            st.subheader("📜 Full Conversation Log")
            for i, msg in enumerate(
                format_history_for_display(chat_history)
            ):
                role = "🧑 You" \
                    if msg["role"] == "user" \
                    else "🤖 Assistant"
                st.markdown(f"**{role}:**")
                st.markdown(msg["content"])
                if i < len(chat_history) - 1:
                    st.divider()