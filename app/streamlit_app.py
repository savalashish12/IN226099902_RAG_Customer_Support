"""
streamlit_app.py — RAG Customer Support Assistant UI
Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from graph.workflow import run_rag_pipeline
from ingestion.load_pdf import load_pdfs
from ingestion.chunking import split_documents
from ingestion.embedding import get_embedding_function
from ingestion.vector_store import store_in_chromadb, collection_exists, delete_collection, list_collections
from app.config import COLLECTION_NAME, LLM_PROVIDER, GROQ_MODEL, GEMINI_MODEL


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; padding: 8px; }
    .confidence-high { color: #22c55e; font-weight: 600; }
    .confidence-low  { color: #f97316; font-weight: 600; }
    .hitl-badge { background: #7c3aed; color: white; padding: 2px 8px;
                  border-radius: 8px; font-size: 0.75rem; }
    .meta-box { background: #1e2130; border-radius: 8px; padding: 10px;
                font-size: 0.8rem; color: #94a3b8; margin-top: 6px; }
    div[data-testid="stSidebar"] { background-color: #161b27; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages"          : [],
        "active_collection" : COLLECTION_NAME,
        "ingested_hash"     : None,    # hash of last ingested file
        "ingestion_done"    : False,   # guard against Streamlit reruns
        "hitl_pending"      : False,
        "hitl_context"      : {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def _ingest_pdf(file_bytes: bytes, file_name: str) -> str:
    """Run ingestion pipeline. Returns collection_name."""
    fhash           = _file_hash(file_bytes)
    collection_name = f"doc_{fhash}"

    # ── KEY FIX: skip if already ingested this exact file ────────────────────
    if collection_exists(collection_name):
        st.sidebar.success(f"✅ Already ingested — reusing collection `{collection_name}`")
        return collection_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with st.sidebar:
            with st.status("Processing document...", expanded=True) as status:
                st.write("📄 Loading pages...")
                docs   = load_pdfs(file_path=tmp_path)

                st.write("✂️ Chunking & cleaning...")
                chunks = split_documents(docs)

                st.write("🔢 Embedding & storing...")
                emb_fn = get_embedding_function()
                store_in_chromadb(chunks, emb_fn, collection_name=collection_name)

                status.update(label="✅ Ingestion complete!", state="complete")
    finally:
        os.remove(tmp_path)

    return collection_name


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Knowledge Base")
    active_model = GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
    st.caption(f"Provider: **{LLM_PROVIDER.upper()}** · Model: `{active_model}`")
    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if uploaded:
        file_bytes = uploaded.getvalue()
        file_hash  = _file_hash(file_bytes)

        # Only run ingestion once per unique file per session
        if st.session_state.ingested_hash != file_hash:
            col_name = _ingest_pdf(file_bytes, uploaded.name)
            st.session_state.active_collection = col_name
            st.session_state.ingested_hash     = file_hash
            st.session_state.ingestion_done    = True
            st.session_state.messages          = [
                {"role": "assistant", "content": f"📄 **{uploaded.name}** ingested! Ask me anything about it."}
            ]
        else:
            st.sidebar.info(f"Using collection: `{st.session_state.active_collection}`")

    st.divider()

    # ── Collection selector ───────────────────────────────────────────────────
    st.subheader("🗂 Active Collections")
    collections = list_collections()
    if collections:
        chosen = st.selectbox(
            "Switch collection:",
            options=collections,
            index=collections.index(st.session_state.active_collection)
                  if st.session_state.active_collection in collections else 0,
            key="collection_selector",
        )
        if chosen != st.session_state.active_collection:
            st.session_state.active_collection = chosen
            st.session_state.messages          = []
            st.rerun()

        # Delete button
        if st.button("🗑 Delete selected collection", use_container_width=True):
            delete_collection(chosen)
            remaining = [c for c in collections if c != chosen]
            st.session_state.active_collection = remaining[0] if remaining else COLLECTION_NAME
            st.session_state.messages          = []
            st.rerun()
    else:
        st.info("No collections yet. Upload a PDF to start.")

    st.divider()
    st.caption(f"Active: `{st.session_state.active_collection}`")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🤖 RAG Customer Support Assistant")
st.caption("Powered by ChromaDB · LangGraph · Gemini Embeddings")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("meta"):
            meta = msg["meta"]
            conf_cls = "confidence-high" if meta["confident"] else "confidence-low"
            conf_label = "High" if meta["confident"] else "Low"

            hitl_badge = (
                '<span class="hitl-badge">HITL Triggered</span>'
                if meta["hitl"] else ""
            )

            st.markdown(
                f'<div class="meta-box">'
                f'Confidence: <span class="{conf_cls}">{conf_label}</span> '
                f'(avg score: {meta["score"]:.4f}) {hitl_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Retrieved Chunks"):
                for i, chunk in enumerate(meta.get("chunks", [])):
                    st.text(f"[{i+1}] {chunk[:300]}...")


# ── HITL override input ───────────────────────────────────────────────────────
if st.session_state.hitl_pending:
    st.warning("⚠️ Low confidence detected. You can override the AI answer below.")
    with st.form("hitl_form"):
        override_text = st.text_area(
            "Human override answer (leave blank to keep AI answer):",
            height=100,
        )
        submitted = st.form_submit_button("Submit Override")
        if submitted:
            ctx = st.session_state.hitl_context
            final_answer = override_text.strip() or ctx["ai_answer"]

            st.session_state.messages.append({
                "role"   : "assistant",
                "content": final_answer,
                "meta"   : {**ctx["meta"], "hitl": True},
            })
            st.session_state.hitl_pending = False
            st.session_state.hitl_context = {}
            st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.active_collection:
        st.error("Please upload a PDF first.")
        st.stop()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_rag_pipeline(
                query           = prompt,
                collection_name = st.session_state.active_collection,
            )

        answer       = result.get("answer", "No answer generated.")
        is_confident = result.get("is_confident", False)
        hitl_needed  = result.get("hitl_needed", False)
        scores       = result.get("scores", [])
        avg_score    = sum(scores) / len(scores) if scores else 0.0
        contexts     = result.get("context", [])

        meta = {
            "confident": is_confident,
            "hitl"     : hitl_needed,
            "score"    : avg_score,
            "chunks"   : contexts,
        }

        if hitl_needed:
            # Don't append to history yet — wait for human override
            st.session_state.hitl_pending = True
            st.session_state.hitl_context = {
                "ai_answer": answer,
                "meta"     : meta,
            }
            st.warning(f"⚠️ Low confidence (score: {avg_score:.4f}). Scroll up to override.")
            st.markdown(f"**AI Best Attempt:** {answer}")
        else:
            st.markdown(answer)
            conf_cls   = "confidence-high" if is_confident else "confidence-low"
            conf_label = "High" if is_confident else "Low"
            st.markdown(
                f'<div class="meta-box">Confidence: '
                f'<span class="{conf_cls}">{conf_label}</span> '
                f'(avg score: {avg_score:.4f})</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role"   : "assistant",
                "content": answer,
                "meta"   : meta,
            })
