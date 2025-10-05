from __future__ import annotations

import os
import sys
import tempfile
from typing import Optional

import streamlit as st

# Support running as a script (streamlit) without a package parent
try:
    from .loader_api import load_api_document
    from .loader_textbook import load_textbook_chapter
    from .loader_paper import load_research_paper
    from .retriever import TfidfRetriever, EmbeddingsRetriever
    from .common import write_json, fetch_url_text
    from .main import maybe_answer_with_llm
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from app.loader_api import load_api_document
    from app.loader_textbook import load_textbook_chapter
    from app.loader_paper import load_research_paper
    from app.retriever import TfidfRetriever, EmbeddingsRetriever
    from app.common import write_json, fetch_url_text
    from app.main import maybe_answer_with_llm


def save_uploaded_pdf(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    try:
        suffix = ".pdf" if not uploaded_file.name.lower().endswith(".pdf") else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            return tmp.name
    except Exception:
        return None


def render_ui() -> None:
    st.set_page_config(page_title="Knowledge Source System", layout="wide")
    st.title("Knowledge Source Processing System - Omer Ben Simon")
    st.caption("Process API docs, textbooks, and research papers (PDF or URL), then ask questions.")
    st.markdown("---")
    st.caption("Made by Omer Ben Simon for Spacial AI")

    with st.sidebar:
        st.header("Input")
        doc_type = st.selectbox("Document type", ["api", "textbook", "paper"], index=0)
        use_llm = st.checkbox("Use LLM", value=True, help="Requires OPENAI_API_KEY")
        api_key_input = st.text_input("OpenAI API Key", type="password", help="Optional override for this session")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
        score_threshold = st.slider("Result threshold", 0.0, 0.2, 0.02, 0.01, help="Hide weak matches below this score.")
        top_k = st.slider("Top K", 1, 10, 5)
        show_sources = st.checkbox("Show sources", value=True)
        retriever_mode = st.radio("Retriever", ["TF-IDF", "Embeddings"], index=1, help="Embeddings needs OPENAI_API_KEY")

    # Initialize session state
    if "index" not in st.session_state:
        st.session_state.index = None
    if "doc_type" not in st.session_state:
        st.session_state.doc_type = None
    if "source" not in st.session_state:
        st.session_state.source = None
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (role, content)

    # Input tabs for better UX
    st.subheader("Provide a source")
    tab_url, tab_pdf = st.tabs(["URL", "PDF"])
    url_value: Optional[str] = None
    pdf_upload = None

    with tab_url:
        url_value = st.text_input("Enter a web URL (HTML or direct PDF)")
    with tab_pdf:
        pdf_upload = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")

    # Build index button
    if st.button("Build/Refresh Index"):
        source: Optional[str] = None
        built = False

        if url_value:
            source = url_value.strip()
            built = True
            # Use loader path below
        elif pdf_upload is not None:
            source = save_uploaded_pdf(pdf_upload)
            if source is None:
                st.error("Failed to read uploaded PDF.")
                return
            built = True
        

        if not built:
            st.error("Provide a URL, PDF, or images to build the index.")
        elif source:
            if doc_type == "api":
                index_path = os.path.join(os.path.dirname(__file__), "..", "data", "api_index.json")
                loader = load_api_document
            elif doc_type == "textbook":
                index_path = os.path.join(os.path.dirname(__file__), "..", "data", "textbook_index.json")
                loader = load_textbook_chapter
            else:
                index_path = os.path.join(os.path.dirname(__file__), "..", "data", "paper_index.json")
                loader = load_research_paper

            with st.spinner("Building index..."):
                index = loader(source)
                write_json(index_path, index)
                st.session_state.index = index
                st.session_state.doc_type = doc_type
                st.session_state.source = source
                st.success("Index ready.")

    # Chat interface
    st.subheader("Chat")
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.write(content)

    user_input = st.chat_input("Ask a question…")
    if user_input:
        # Immediately render the user's message in the chat
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.history.append(("user", user_input))
        answer_text = ""
        with st.chat_message("assistant"):
            if not st.session_state.index:
                answer_text = "Please build the index first using the button on the left."
                st.write(answer_text)
            else:
                if retriever_mode == "Embeddings":
                    retriever = EmbeddingsRetriever()
                else:
                    retriever = TfidfRetriever()
                retriever.fit(st.session_state.index)
                results = retriever.query(user_input, top_k=top_k)

                # Decide context
                context = ""
                max_score = max((r.score for r in results), default=0.0)
                strong_results = [r for r in results if r.score >= score_threshold]
                show_local = len(strong_results) > 0

                if use_llm:
                    try_raw = False
                    if st.session_state.source and isinstance(st.session_state.source, str) and st.session_state.source.lower().startswith("http"):
                        try_raw = max_score < score_threshold
                    if try_raw:
                        raw_text = fetch_url_text(st.session_state.source) or ""
                        if raw_text.strip():
                            context = raw_text[:12000]
                    if not context:
                        base_chunks = strong_results if strong_results else results
                        context = "\n\n".join(r.chunk for r in base_chunks)

                # Generate answer or fallback
                if use_llm and context.strip():
                    with st.spinner("Generating answer..."):
                        ans = maybe_answer_with_llm(context, user_input)
                    if ans:
                        answer_text = ans
                        st.write(ans)
                if not answer_text:
                    if show_local:
                        best = max(strong_results, key=lambda r: r.score)
                        answer_text = best.chunk[:1200]
                        st.write(answer_text)
                    else:
                        answer_text = "No strong matches found. Try a more specific question or enable LLM."
                        st.write(answer_text)

                # Sources expander
                if show_sources:
                    with st.expander("Sources"):
                        if strong_results:
                            for r in strong_results:
                                st.markdown(f"**score:** {r.score:.3f}")
                                st.code(r.chunk[:800])
                        else:
                            st.write("No sources above threshold.")

        st.session_state.history.append(("assistant", answer_text))


if __name__ == "__main__":
    render_ui()


