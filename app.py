import html
import os

import markdown as md_lib

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI Assistant",
    page_icon="DM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --bg: #0f172a;
  --card: #1e293b;
  --input: #111827;
  --accent: #3b82f6;
  --text-primary: #e5e7eb;
  --text-secondary: #9ca3af;
  --border: #334155;
  --success: #10b981;
}

.stApp {
  background: var(--bg);
  color: var(--text-primary);
}

[data-testid="stSidebar"] {
  background: #0b1220;
  border-right: 1px solid #1f2937;
}

[data-testid="stSidebar"] * {
  color: var(--text-primary);
}

.side-card {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 14px;
  padding: 12px;
}

.status-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid #334155;
}

.status-ok {
  color: #6ee7b7;
  background: rgba(16, 185, 129, 0.10);
  border-color: rgba(16, 185, 129, 0.35);
}

.status-bad {
  color: #fbbf24;
  background: rgba(245, 158, 11, 0.10);
  border-color: rgba(245, 158, 11, 0.35);
}

.main-shell {
  max-width: 1080px;
  margin: 0 auto;
}

.upload-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
}

.upload-title {
  margin: 0 0 4px 0;
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
}

.upload-subtitle {
  margin: 0 0 14px 0;
  color: var(--text-secondary);
  font-size: 14px;
}

.file-preview {
  margin-top: 12px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--input);
  padding: 10px 12px;
}

.file-name {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 14px;
}

.file-meta {
  font-size: 12px;
  color: var(--text-secondary);
}

.chat-shell {
  margin-top: 16px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
}

.chat-header {
  margin: 0 0 10px 2px;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-secondary);
}

.chat-scroll {
  height: 52vh;
  overflow-y: auto;
  border: 1px solid #263244;
  border-radius: 14px;
  background: #0b1220;
  padding: 14px;
}

.msg-row {
  display: flex;
  width: 100%;
  margin: 0 0 12px 0;
}

.msg-row.user {
  justify-content: flex-end;
}

.msg-row.assistant {
  justify-content: flex-start;
}

.msg-bubble {
  max-width: 78%;
  padding: 10px 12px;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.6;
  white-space: normal;
}

.msg-bubble.user {
  background: #2563eb;
  color: white;
  border-bottom-right-radius: 6px;
}

.msg-bubble.assistant {
  background: #111827;
  color: var(--text-primary);
  border: 1px solid #334155;
  border-bottom-left-radius: 6px;
}

.source-wrap {
  margin-top: 8px;
}

.source-item {
  border: 1px solid #334155;
  background: #0f172a;
  border-radius: 10px;
  padding: 8px 10px;
  margin-top: 6px;
}

.source-head {
  font-size: 12px;
  color: #93c5fd;
  margin: 0 0 4px 0;
  font-weight: 600;
}

.source-body {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.empty-chat {
  text-align: center;
  color: var(--text-secondary);
  padding: 32px 10px;
  font-size: 14px;
}

.ready-banner {
  margin-top: 12px;
  border: 1px solid rgba(16, 185, 129, 0.35);
  background: rgba(16, 185, 129, 0.12);
  color: #86efac;
  border-radius: 12px;
  padding: 10px 12px;
  font-size: 14px;
}

div.stButton > button {
  border-radius: 12px;
  border: 1px solid #3b82f6;
  background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  font-weight: 600;
  transition: all 0.2s ease;
}

div.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 20px rgba(37, 99, 235, 0.25);
}

[data-testid="stFileUploader"] > div {
  border-radius: 12px;
  border: 1px dashed #475569;
  background: #111827;
}

[data-testid="stChatInput"] {
  background: rgba(15, 23, 42, 0.96);
  border-top: 1px solid #1f2937;
}

[data-testid="stChatInput"] textarea {
  background: #111827 !important;
  color: var(--text-primary) !important;
  border: 1px solid #334155 !important;
  border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: #9ca3af !important;
}

[data-testid="stMarkdownContainer"] p {
  margin-bottom: 0.3rem;
}

/* Markdown rendering inside assistant bubbles */
.msg-bubble.assistant p {
  margin: 0 0 8px 0;
  line-height: 1.7;
}

.msg-bubble.assistant p:last-child {
  margin-bottom: 0;
}

.msg-bubble.assistant ul,
.msg-bubble.assistant ol {
  margin: 6px 0 8px 0;
  padding-left: 20px;
}

.msg-bubble.assistant li {
  margin-bottom: 4px;
  line-height: 1.6;
}

.msg-bubble.assistant li > ul,
.msg-bubble.assistant li > ol {
  margin-top: 4px;
  margin-bottom: 0;
}

.msg-bubble.assistant strong {
  color: #e2e8f0;
  font-weight: 700;
}

.msg-bubble.assistant em {
  color: #93c5fd;
}

.msg-bubble.assistant h1,
.msg-bubble.assistant h2,
.msg-bubble.assistant h3,
.msg-bubble.assistant h4 {
  color: #e2e8f0;
  margin: 10px 0 6px 0;
  font-weight: 700;
  line-height: 1.3;
}

.msg-bubble.assistant h1 { font-size: 16px; }
.msg-bubble.assistant h2 { font-size: 15px; }
.msg-bubble.assistant h3 { font-size: 14px; }

.msg-bubble.assistant code {
  background: #1e3a5f;
  color: #93c5fd;
  padding: 2px 5px;
  border-radius: 4px;
  font-size: 12px;
  font-family: monospace;
}

.msg-bubble.assistant blockquote {
  border-left: 3px solid #3b82f6;
  margin: 6px 0;
  padding: 4px 10px;
  color: #9ca3af;
}

.msg-bubble.assistant hr {
  border: none;
  border-top: 1px solid #334155;
  margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state():
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = {}


def safe_text(text: str) -> str:
    return html.escape(str(text)).replace("\n", "<br>")


def format_size(file_size: int) -> str:
    if file_size < 1024:
        return f"{file_size} B"
    if file_size < 1024 * 1024:
        return f"{file_size / 1024:.1f} KB"
    return f"{file_size / (1024 * 1024):.1f} MB"


def render_markdown(text: str) -> str:
    """Convert markdown text to safe HTML for rendering inside chat bubbles."""
    return md_lib.markdown(
        text,
        extensions=["nl2br", "sane_lists"],
    )


def ChatMessage(role: str, content: str, sources=None):
    role_class = "user" if role == "user" else "assistant"

    # User messages: escape HTML for safety
    # Assistant messages: render markdown to HTML so **bold**, bullets, etc. display correctly
    if role == "assistant":
        rendered_content = render_markdown(content)
    else:
        rendered_content = safe_text(content)

    bubble = (
        f'<div class="msg-row {role_class}">'
        f'<div class="msg-bubble {role_class}">{rendered_content}</div>'
        "</div>"
    )

    if role != "assistant" or not sources:
        return bubble

    source_parts = ['<div class="msg-row assistant"><div class="source-wrap">']
    for i, src in enumerate(sources, start=1):
        page = html.escape(str(src.get("page", "?")))
        score = src.get("similarity_score", "")
        score_text = f" | relevance {score}" if score != "" else ""
        raw_snippet = str(src.get("text", ""))[:320].strip()
        ellipsis = "..." if len(str(src.get("text", ""))) > 320 else ""
        snippet = safe_text(raw_snippet)
        source_parts.append(
            '<div class="source-item">'
            f'<div class="source-head">Source {i} | Page {page}{score_text}</div>'
            f'<div class="source-body">{snippet}{ellipsis}</div>'
            "</div>"
        )
    source_parts.append("</div></div>")
    return bubble + "".join(source_parts)


def UploadCard(uploaded_file):
    st.markdown(
        """
<div class="upload-card">
  <h1 class="upload-title">DocuMind AI Assistant</h1>
  <p class="upload-subtitle">Upload a document, process it, then chat with your PDF in dark mode.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    file_obj = st.file_uploader(
        "Upload PDF (drag and drop supported)",
        type=["pdf"],
        help="Text-based PDFs work best.",
    )

    if file_obj:
        st.markdown(
            f"""
<div class="file-preview">
  <div class="file-name">{safe_text(file_obj.name)}</div>
  <div class="file-meta">{format_size(file_obj.size)} | PDF</div>
</div>
""",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c2:
        process_clicked = st.button(
            "Process Document",
            use_container_width=True,
            key="process_document_main",
        )

    return file_obj or uploaded_file, process_clicked


def Sidebar(api_key: str):
    with st.sidebar:
        st.markdown("## DocuMind")
        st.caption("Project controls")
        st.divider()

        entered_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=api_key,
            placeholder="AIza...",
        )

        connected = bool((entered_key or "").strip())
        if connected:
            os.environ["GOOGLE_API_KEY"] = entered_key

        status_class = "status-ok" if connected else "status-bad"
        status_text = "Connected" if connected else "Not connected"
        st.markdown(
            f'<div class="side-card"><span class="status-pill {status_class}">{status_text}</span></div>',
            unsafe_allow_html=True,
        )

        if st.session_state.doc_stats:
            stats = st.session_state.doc_stats
            st.markdown("### Document Stats")
            st.markdown(
                f"""
<div class="side-card">
  <div>Pages: <strong>{stats.get("pages", 0)}</strong></div>
  <div>Chunks: <strong>{stats.get("chunks", 0)}</strong></div>
  <div>Words: <strong>{stats.get("words", 0):,}</strong></div>
</div>
""",
                unsafe_allow_html=True,
            )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    return entered_key


def process_document(uploaded_file, api_key):
    with st.spinner("Extracting text, creating embeddings, and indexing..."):
        rag = RAGEngine(api_key=api_key)
        stats = rag.load_pdf(uploaded_file)
        st.session_state.rag = rag
        st.session_state.doc_stats = stats
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Document is ready. Ask your questions about the document.",
                "sources": [],
            }
        ]


def process_question(question: str):
    clean_q = (question or "").strip()
    if not clean_q:
        return

    if not st.session_state.rag:
        st.warning("Process a document first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": clean_q, "sources": []})
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.rag.query(
                        clean_q,
                        chat_history=st.session_state.chat_history[:-1],  # exclude current user message
                    )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "sources": result.get("sources", []),
                }
            )
        except Exception as exc:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Sorry, I hit an error: {exc}",
                    "sources": [],
                }
            )


def render_chat_area():
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">Chat</div>', unsafe_allow_html=True)

    if not st.session_state.rag and not st.session_state.chat_history:
        st.markdown(
            '<div class="chat-scroll"><div class="empty-chat">Process a PDF to start chatting.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    parts = ['<div class="chat-scroll">']
    for msg in st.session_state.chat_history:
        parts.append(ChatMessage(msg.get("role", "assistant"), msg.get("content", ""), msg.get("sources")))
    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


init_state()

api_key = Sidebar(os.getenv("GOOGLE_API_KEY", ""))

st.markdown('<div class="main-shell">', unsafe_allow_html=True)

current_upload = None
uploaded_file, process_clicked = UploadCard(current_upload)

if process_clicked:
    if not (api_key or "").strip():
        st.error("Please provide Gemini API key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a PDF first.")
    else:
        try:
            process_document(uploaded_file, api_key)
            st.markdown(
                '<div class="ready-banner">Success: Document processed. Chat input is now enabled.</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.session_state.rag = None
            st.session_state.doc_stats = {}
            st.error(f"Processing failed: {exc}")

render_chat_area()

st.markdown("</div>", unsafe_allow_html=True)

user_question = st.chat_input(
    "Ask your questions about the document...",
    disabled=not bool(st.session_state.rag),
)

if user_question:
    process_question(user_question)
    st.rerun()

