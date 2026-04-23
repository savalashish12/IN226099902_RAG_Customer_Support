"""
streamlit_app.py — Enterprise RAG Customer Support Assistant UI

Features:
  ✅ PDF upload with MD5 dedup (no re-embedding same file)
  ✅ Multi-collection switching (HR / IT / Customer Support)
  ✅ Real-time confidence meter (High/Medium/Low with similarity %)
  ✅ Retrieved chunks expander (transparency)
  ✅ HITL panel: Approve or Override AI answer
  ✅ HITL uses real LangGraph interrupt() + Command(resume=...)
  ✅ Escalation history log in sidebar
  ✅ Chat history within session
"""
import streamlit as st
import uuid
import hashlib

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNova Support Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingestion.load_pdf import load_and_ingest_pdf
from ingestion.vector_store import (
    list_collections, delete_collection, collection_exists
)
from graph.workflow import run_rag_pipeline, resume_with_human_input, get_graph_state
from hitl.manager import hitl_manager

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "active_collection" : None,
        "collection_label"  : None,
        "chat_history"      : [],       # list of {role, content, meta}
        "pending_hitl"      : None,     # {thread_id, query, ai_answer, ...}
        "session_id"        : str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_confidence_ui(confidence: str, similarity_pct: float) -> tuple[str, str]:
    """Returns (badge_html, color_hex) for confidence display."""
    if confidence == "HIGH":
        badge = f"🟢 High Confidence &nbsp;|&nbsp; {similarity_pct:.0f}% Match"
        color = "#1a7a4a"
    elif confidence == "MEDIUM":
        badge = f"🟡 Medium Confidence &nbsp;|&nbsp; {similarity_pct:.0f}% Match"
        color = "#b8860b"
    else:
        badge = f"🔴 Low Confidence &nbsp;|&nbsp; {similarity_pct:.0f}% Match — Escalated to Human"
        color = "#c0392b"
    return badge, color


def get_file_md5(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 TechNova Support")
    st.markdown("---")

    # ── PDF Upload ────────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base")
    st.caption("Upload a company PDF to begin")

    uploaded = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded:
        file_bytes = uploaded.read()
        md5        = get_file_md5(file_bytes)
        col_name   = f"doc_{md5}"

        if not collection_exists(col_name):
            with st.spinner(f"📄 Ingesting {uploaded.name}..."):
                try:
                    count = load_and_ingest_pdf(file_bytes, col_name, uploaded.name)
                    st.success(f"✅ Ingested {count} chunks")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
        else:
            st.info(f"♻️ Using cached: {uploaded.name}")

        st.session_state["active_collection"] = col_name
        st.session_state["collection_label"]  = uploaded.name

    # ── Collection Switcher ────────────────────────────────────
    st.markdown("### 🗂️ Collections")
    collections = list_collections()

    if collections:
        for col in collections:
            is_active = col == st.session_state.get("active_collection")
            btn_label = f"{'▶ ' if is_active else ''}{col}"
            
            c1, c2 = st.columns([4, 1])
            with c1:
                if st.button(btn_label, key=f"col_{col}", use_container_width=True):
                    st.session_state["active_collection"] = col
                    st.session_state["collection_label"]  = col
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{col}", help="Delete this collection"):
                    delete_collection(col)
                    if is_active:
                        st.session_state["active_collection"] = None
                        st.session_state["collection_label"] = None
                    st.rerun()
    else:
        st.caption("No collections yet. Upload a PDF.")

    st.markdown("---")

    # ── HITL History ──────────────────────────────────────────
    st.markdown("### 📋 Escalation Log")
    history = hitl_manager.get_history()
    if history:
        for h in reversed(history[-5:]):
            status_icon = {"PENDING": "⏳", "APPROVED": "✅", "OVERRIDDEN": "✏️"}.get(h["status"], "❓")
            with st.expander(f"{status_icon} {h['query'][:40]}..."):
                st.caption(f"Status: **{h['status']}**")
                st.caption(f"Reason: {h['reason']}")
                st.caption(f"Created: {h['created_at'][:19]}")
    else:
        st.caption("No escalations yet.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
active_col   = st.session_state.get("active_collection")
active_label = st.session_state.get("collection_label", active_col or "")

st.markdown(f"# 🖥️ TechNova Internal Support Assistant")
st.markdown(
    f"**Active KB:** `{active_label or 'None — upload a PDF to begin'}`"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"**LLM:** Groq · llama-3.3-70b-versatile"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"**Vector DB:** ChromaDB"
)
st.markdown("---")

with st.expander("🤔 How does this work? (Simple Explanation)"):
    st.markdown("""
    **Hi! I am your AI Assistant. Here's how to use me:**
    1. 📄 **Upload a PDF:** Click the upload button on the left to give me a document to read.
    2. 💬 **Ask a Question:** Type your question in the chat box at the bottom.
    3. 🤖 **I Search the Document:** I will read through the PDF you gave me to find the exact answer.
    4. 🚦 **Confidence Check:** 
       - 🟢 **High Match:** I am sure about the answer!
       - 🟡 **Medium Match:** I found some information, but it might not be perfect.
       - 🔴 **Low Match:** I couldn't find the answer, so I will ask a human to review it!
    """)

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            confidence = meta.get("confidence_level", "LOW")
            similarity = meta.get("similarity_pct", 0.0)
            badge, color = get_confidence_ui(confidence, similarity)

            st.markdown(
                f'<span style="color:{color}; font-size:0.85em; font-weight:600">'
                f'{badge}</span>',
                unsafe_allow_html=True
            )

            # Retrieved chunks expander
            if meta.get("context"):
                with st.expander(f"🔍 {len(meta['context'])} Retrieved Chunks", expanded=False):
                    for i, chunk in enumerate(meta["context"]):
                        score = meta["scores"][i] if i < len(meta.get("scores", [])) else "—"
                        sim   = round((1 - score) * 100, 1) if isinstance(score, float) else "—"
                        st.markdown(f"**Chunk {i+1}** — L2 Score: `{score}` | Similarity: `{sim}%`")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.divider()


# ── HITL Panel ────────────────────────────────────────────────────────────────
pending = st.session_state.get("pending_hitl")

if pending:
    with st.container():
        st.markdown("---")
        st.markdown(
            "### 🚨 Human Review Required",
            help="The AI could not answer this with high confidence. Please review."
        )

        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.error(f"**Escalation Reason:** {pending.get('reason', 'Low confidence')}")
            st.markdown(f"**User Query:** {pending['query']}")
            st.markdown("**AI-Generated Answer:**")
            st.info(pending["ai_answer"])

        with col_r:
            st.markdown("**Your Decision:**")
            override_text = st.text_area(
                "Override answer (leave blank to approve AI answer):",
                height=150,
                placeholder="Type a better answer here, or leave blank to approve...",
                key="hitl_override_text"
            )

            c1, c2 = st.columns(2)

            with c1:
                if st.button("✅ Approve AI Answer", use_container_width=True, type="secondary"):
                    result = resume_with_human_input(
                        thread_id       = pending["thread_id"],
                        approved_answer = pending["ai_answer"],
                        overridden      = False,
                    )
                    hitl_manager.resolve(pending["thread_id"], pending["ai_answer"], False)

                    # Add resolved answer to chat
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": f"✅ **[Human Approved]**\n\n{pending['ai_answer']}",
                        "meta": pending.get("meta", {})
                    })
                    st.session_state["pending_hitl"] = None
                    st.rerun()

            with c2:
                if st.button("✏️ Submit Override", use_container_width=True, type="primary"):
                    final_answer = override_text.strip() or pending["ai_answer"]
                    overridden   = bool(override_text.strip())

                    result = resume_with_human_input(
                        thread_id       = pending["thread_id"],
                        approved_answer = final_answer,
                        overridden      = overridden,
                    )
                    hitl_manager.resolve(pending["thread_id"], final_answer, overridden)

                    label = "Human Overridden" if overridden else "Human Approved"
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": f"{'✏️' if overridden else '✅'} **[{label}]**\n\n{final_answer}",
                        "meta": pending.get("meta", {})
                    })
                    st.session_state["pending_hitl"] = None
                    st.rerun()

        st.markdown("---")


# ── Chat Input ────────────────────────────────────────────────────────────────
if not active_col:
    st.warning("⬅️ Please upload a PDF from the sidebar to start.")
else:
    user_input = st.chat_input("Ask a question about your company policy or IT issues...")

    if user_input:
        # Don't process new queries while HITL is pending
        if pending:
            st.warning("⚠️ Please resolve the pending human review before asking a new question.")
        else:
            # Show user message
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Run RAG pipeline
            thread_id = f"thread_{st.session_state['session_id']}_{uuid.uuid4().hex[:8]}"

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base..."):
                    try:
                        result = run_rag_pipeline(
                            query           = user_input,
                            collection_name = active_col,
                            thread_id       = thread_id,
                        )

                        answer           = result.get("answer", "No answer generated.")
                        confidence_level = result.get("confidence_level", "LOW")
                        similarity_pct   = result.get("similarity_pct", 0.0)
                        hitl_needed      = result.get("hitl_needed", False)
                        context          = result.get("context", [])
                        scores           = result.get("scores", [])
                        hitl_reason      = result.get("hitl_reason", "")

                        badge, color = get_confidence_ui(confidence_level, similarity_pct)

                        if hitl_needed:
                            # Store pending HITL — do NOT show answer yet
                            hitl_session_id = hitl_manager.create_session(
                                query      = user_input,
                                ai_answer  = answer,
                                confidence = confidence_level,
                                reason     = hitl_reason,
                            )
                            st.session_state["pending_hitl"] = {
                                "thread_id" : thread_id,
                                "query"     : user_input,
                                "ai_answer" : answer,
                                "reason"    : hitl_reason,
                                "confidence": confidence_level,
                                "meta"      : {
                                    "confidence_level": confidence_level,
                                    "similarity_pct"  : similarity_pct,
                                    "context"         : context,
                                    "scores"          : scores,
                                }
                            }
                            st.warning(
                                f"🚨 This query has been escalated for human review.\n\n"
                                f"**Reason:** {hitl_reason}"
                            )
                        else:
                            # Show answer directly
                            st.markdown(answer)
                            st.markdown(
                                f'<span style="color:{color}; font-size:0.85em; font-weight:600">'
                                f'{badge}</span>',
                                unsafe_allow_html=True
                            )

                            # Retrieved chunks
                            if context:
                                with st.expander(f"🔍 {len(context)} Retrieved Chunks", expanded=False):
                                    for i, chunk in enumerate(context):
                                        score = scores[i] if i < len(scores) else None
                                        sim   = round((1 - score) * 100, 1) if score is not None else "—"
                                        st.markdown(f"**Chunk {i+1}** | L2 Score: `{round(score,4) if score else '—'}` | Similarity: `{sim}%`")
                                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                        st.divider()

                            # Save to history
                            st.session_state["chat_history"].append({
                                "role"   : "assistant",
                                "content": answer,
                                "meta"   : {
                                    "confidence_level": confidence_level,
                                    "similarity_pct"  : similarity_pct,
                                    "context"         : context,
                                    "scores"          : scores,
                                }
                            })

                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Pipeline error: {e}")
                        st.exception(e)
