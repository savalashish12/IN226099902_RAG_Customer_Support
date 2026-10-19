"""
streamlit_app.py — Enterprise RAG Customer Support Assistant UI

Features:
  ✅ Batch PDF upload: multiple files / local folder / ZIP archive
  ✅ Per-file MD5 dedup (no re-embedding same file)
  ✅ Multi-collection switching (HR / IT / Customer Support)
  ✅ Real-time confidence meter (High/Medium/Low with similarity %)
  ✅ Retrieved chunks expander (transparency)
  ✅ HITL panel: Approve or Override AI answer
  ✅ HITL uses real LangGraph interrupt() + Command(resume=...)
  ✅ Escalation history log in sidebar
  ✅ Chat history within session
"""
import hashlib
import json
import uuid

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechNova Support Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import database as db
from graph.nodes import _classify_confidence
from graph.workflow import (
    get_graph_state,
    resume_with_human_input,
    run_rag_pipeline,
    run_retrieval_only,
)
from hitl.manager import hitl_manager
from ingestion import worker as ingest_worker
from ingestion.batch import (
    discover_pdfs_in_folder,
    extract_pdfs_from_zip,
    get_file_md5,
    ingest_batch,
    ingest_one,
)
from ingestion.vector_store import collection_exists, delete_collection, list_collections
from llm.groq_client import stream_answer

db.init_db()

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
        "ingest_job_id"     : None,     # active background ingestion job id
        "ingest_progress"   : [],       # [{name, status, chunks, collection, error, idx}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Restore persisted chat history for this session (survives reloads)
    if not st.session_state.get("_history_loaded"):
        try:
            rows = db.get_messages(st.session_state["session_id"])
            restored = []
            for r in rows:
                meta = {}
                if r["role"] == "assistant":
                    meta = {
                        "confidence_level": r["confidence"],
                        "similarity_pct"  : r["similarity_pct"],
                        "sources"         : json.loads(r["sources"]) if r["sources"] else [],
                        "context"         : [],
                        "scores"          : [],
                    }
                restored.append({"role": r["role"], "content": r["content"], "meta": meta})
            if restored:
                st.session_state["chat_history"] = restored
        except Exception:
            pass
        st.session_state["_history_loaded"] = True

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


_ESCALATION_KEYWORDS = [
    "urgent", "critical", "emergency", "not working", "failed",
    "error", "broken", "crash", "data loss", "security breach",
]


def _evaluate_local(avg_score: float, query: str, answer: str) -> tuple[str, bool, str]:
    """
    Mirror of evaluation_node's HITL logic for the UI streaming path.
    Returns (confidence_level, hitl_needed, hitl_reason).
    """
    confidence_level, _is_confident, hitl_needed, hitl_reason = \
        _classify_confidence(avg_score, answer)

    if confidence_level == "MEDIUM":
        if any(kw in query.lower() for kw in _ESCALATION_KEYWORDS):
            hitl_needed = True
            hitl_reason = "Urgent/critical keyword detected"
        elif len(query.split()) > 10:
            hitl_needed = True
            hitl_reason = "Complex multi-part query"

    return confidence_level, hitl_needed, hitl_reason


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 TechNova Support")
    st.markdown("---")

    # ── PDF Upload (Batch) ─────────────────────────────────────
    st.markdown("### 📚 Knowledge Base")
    st.caption("Upload PDFs: multiple files, folder path, or ZIP archive")

    upload_mode = st.radio(
        "Upload Mode",
        options=["Multiple Files", "Folder Path (local)", "ZIP Archive"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Collect items (name, bytes) to ingest based on the selected mode
    _ingest_items: list[tuple[str, bytes]] = []

    if upload_mode == "Multiple Files":
        uploaded_list = st.file_uploader(
            "Select PDF files", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if st.button("🚀 Ingest Selected Files", disabled=not uploaded_list):
            _ingest_items = [(u.name, u.read()) for u in uploaded_list]

    elif upload_mode == "Folder Path (local)":
        folder = st.text_input("Local folder path", placeholder="e.g. C:/docs/policies")
        if st.button("📁 Discover & Ingest PDFs", disabled=not folder.strip()):
            try:
                _ingest_items = discover_pdfs_in_folder(folder.strip())
                if not _ingest_items:
                    st.warning("No PDF files found in that folder (recursively)")
            except Exception as e:
                st.error(f"❌ {e}")

    else:  # ZIP
        zipup = st.file_uploader("Upload ZIP containing PDFs", type=["zip"],
                                 label_visibility="collapsed")
        if st.button("📦 Extract & Ingest", disabled=not zipup):
            try:
                _ingest_items = extract_pdfs_from_zip(zipup.read())
                if not _ingest_items:
                    st.warning("No PDFs found inside ZIP archive")
            except Exception as e:
                st.error(f"❌ {e}")

    # Launch a background ingestion job if items were collected
    if _ingest_items:
        ingest_worker.prune_jobs(keep=5)
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        st.session_state["ingest_job_id"] = job_id
        st.session_state["ingest_progress"] = []
        ingest_worker.start_job(job_id, _ingest_items)

    # ── Live Ingestion Progress (auto-refreshing) ─────────────
    @st.fragment(run_every=1.5)
    def _render_ingest_progress():
        job_id = st.session_state.get("ingest_job_id")
        if not job_id:
            return
        job = ingest_worker.get_job(job_id)
        if not job:
            return
        snap = job.snapshot()

        st.markdown("#### 📋 Ingestion Progress")
        st.progress(min(1.0, snap["progress"]),
                    text=f"{snap['completed']}/{snap['total']} files")

        for p in snap["results"]:
            if p["status"] == "ingested":
                st.success(f"✅ {p['name']} → {p['chunks']} chunks")
            elif p["status"] == "cached":
                st.info(f"♻️ {p['name']} → cached")
            else:
                st.error(f"❌ {p['name']} → {p['error']}")

        if snap["status"] == "done":
            # Auto-select the first ingested collection once
            if not st.session_state.get("_job_selected") == job_id:
                first = job.first_collection()
                if first:
                    st.session_state["active_collection"] = first
                    for r in snap["results"]:
                        if r.get("collection") == first:
                            st.session_state["collection_label"] = r["name"]
                            break
                st.session_state["_job_selected"] = job_id
            st.caption(f"✅ Ingestion complete ({snap['completed']} files).")
        elif snap["status"] == "error":
            st.error(f"Job failed: {snap['error']}")
        else:
            st.caption("Ingesting in the background — you can keep using the app.")

    _render_ingest_progress()

    # ── Search Scope ───────────────────────────────────────────
    st.markdown("### 🔍 Search Scope")
    search_all = st.toggle(
        "Search ALL documents",
        value=st.session_state.get("search_all", True),
        help="When ON, every ingested document is searched. When OFF, only the active collection is used.",
    )
    st.session_state["search_all"] = search_all

    stream_mode = st.toggle(
        "Stream responses",
        value=st.session_state.get("stream_mode", True),
        help="Show answers token-by-token as they are generated.",
    )
    st.session_state["stream_mode"] = stream_mode

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

st.markdown("# 🖥️ TechNova Internal Support Assistant")
_scope = "All documents" if st.session_state.get("search_all", True) else (active_label or "None")
st.markdown(
    f"**Active KB:** `{active_label or 'None — upload a PDF to begin'}`"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"**Search Scope:** `{_scope}`"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"**LLM:** Groq · openai/gpt-oss-120b"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"**Vector DB:** ChromaDB"
)
st.caption("📊 Admins: open the **Analytics** page from the sidebar for usage, quality, and knowledge-gap insights.")
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

# ── Chat Controls ─────────────────────────────────────────────────────────────
_ctrl_l, _ctrl_r = st.columns([6, 1])
with _ctrl_r:
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            try:
                db.clear_messages(st.session_state["session_id"])
            except Exception:
                pass
            st.session_state["chat_history"] = []
            st.session_state["pending_hitl"] = None
            st.rerun()

# ── Chat History ──────────────────────────────────────────────────────────────
for _idx, msg in enumerate(st.session_state["chat_history"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            confidence = meta.get("confidence_level", "LOW")
            similarity = meta.get("similarity_pct", 0.0)
            badge, color = get_confidence_ui(confidence, similarity)

            # Source citations (filename + page)
            _srcs = meta.get("sources", [])
            if _srcs:
                _cites = []
                for s in _srcs:
                    fn = s.get("source") or s.get("filename") or s.get("collection", "doc")
                    pg = s.get("page")
                    _cites.append(f"{fn}" + (f", p.{pg + 1}" if isinstance(pg, int) else ""))
                _uniq = list(dict.fromkeys(_cites))
                st.markdown(
                    '<span style="font-size:0.8em; color:#555">📎 Sources: '
                    + " · ".join(_uniq) + "</span>",
                    unsafe_allow_html=True
                )

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

            # Feedback (👍 / 👎) — persisted to SQLite
            fb_key = f"fb_{_idx}"
            _prev = st.session_state.get(fb_key)
            fc1, fc2, fc3 = st.columns([1, 1, 8])
            if fc1.button("👍", key=f"up_{_idx}"):
                db.add_feedback("up", query=meta.get("_query", ""),
                                session_id=st.session_state["session_id"])
                st.session_state[fb_key] = "up"
                st.toast("Thanks for your feedback!")
            if fc2.button("👎", key=f"down_{_idx}"):
                db.add_feedback("down", query=meta.get("_query", ""),
                                session_id=st.session_state["session_id"])
                st.session_state[fb_key] = "down"
                st.toast("Thanks — this helps us find knowledge gaps.")
            if _prev:
                fc3.caption(f"Your feedback: {'👍 helpful' if _prev == 'up' else '👎 not helpful'}")


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
                    _content = f"✅ **[Human Approved]**\n\n{pending['ai_answer']}"
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": _content,
                        "meta": pending.get("meta", {})
                    })
                    try:
                        db.add_message(
                            st.session_state["session_id"], "assistant", _content,
                            confidence=pending.get("confidence"),
                            similarity_pct=pending.get("meta", {}).get("similarity_pct"),
                            sources=pending.get("meta", {}).get("sources", []),
                        )
                    except Exception:
                        pass
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
                    _content = f"{'✏️' if overridden else '✅'} **[{label}]**\n\n{final_answer}"
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": _content,
                        "meta": pending.get("meta", {})
                    })
                    try:
                        db.add_message(
                            st.session_state["session_id"], "assistant", _content,
                            confidence=pending.get("confidence"),
                            similarity_pct=pending.get("meta", {}).get("similarity_pct"),
                            sources=pending.get("meta", {}).get("sources", []),
                        )
                    except Exception:
                        pass
                    st.session_state["pending_hitl"] = None
                    st.rerun()

        st.markdown("---")


# ── Chat Input ────────────────────────────────────────────────────────────────
_available_collections = list_collections()
if not _available_collections:
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
            try:
                db.add_message(st.session_state["session_id"], "user", user_input)
            except Exception:
                pass
            with st.chat_message("user"):
                st.markdown(user_input)

            # Run RAG pipeline
            thread_id = f"thread_{st.session_state['session_id']}_{uuid.uuid4().hex[:8]}"

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base..."):
                    try:
                        # Determine search scope: all collections or just active
                        if st.session_state.get("search_all", True):
                            search_cols = list_collections()
                        else:
                            search_cols = [active_col] if active_col else list_collections()

                        # Prior turns (role + content only) for conversational memory
                        _hist = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_history"][:-1]
                            if m["role"] in ("user", "assistant")
                        ]

                        _used_graph_path = False

                        # ── Streaming path (fast, non-HITL) ──────────
                        if st.session_state.get("stream_mode", True):
                            prep = run_retrieval_only(
                                query            = user_input,
                                collection_name  = active_col,
                                collection_names = search_cols,
                                chat_history     = _hist,
                            )
                            context  = prep["context"]
                            scores   = prep["scores"]
                            sources  = prep["sources"]
                            avg      = prep["avg_score"]
                            sim_pct  = round(max(0.0, (1.0 - avg) * 100), 1)

                            # If retrieval is clearly out of scope → go through
                            # the full graph to trigger the proper HITL flow.
                            if prep["guard_no_answer"] or not context:
                                _used_graph_path = True
                                result = run_rag_pipeline(
                                    query=user_input, collection_name=active_col,
                                    collection_names=search_cols,
                                    thread_id=thread_id, chat_history=_hist,
                                )
                                answer           = result.get("answer", "")
                                confidence_level = result.get("confidence_level", "LOW")
                                similarity_pct   = result.get("similarity_pct", 0.0)
                                hitl_needed      = result.get("hitl_needed", True)
                                context          = result.get("context", [])
                                scores           = result.get("scores", [])
                                sources          = result.get("sources", [])
                                hitl_reason      = result.get("hitl_reason", "Query out of document scope")
                            else:
                                # Stream generation directly
                                with st.spinner("✍️ Generating answer..."):
                                    answer = st.write_stream(
                                        stream_answer(
                                            user_input, context,
                                            instruction="",
                                            sources=sources,
                                        )
                                    )
                                answer = answer or ""
                                # Local evaluation (mirrors evaluation_node)
                                confidence_level, hitl_needed, hitl_reason = _evaluate_local(
                                    avg, user_input, answer
                                )
                                similarity_pct = sim_pct
                        else:
                            result = run_rag_pipeline(
                                query            = user_input,
                                collection_name  = active_col,
                                collection_names = search_cols,
                                thread_id        = thread_id,
                                chat_history     = _hist,
                            )
                            answer           = result.get("answer", "No answer generated.")
                            confidence_level = result.get("confidence_level", "LOW")
                            similarity_pct   = result.get("similarity_pct", 0.0)
                            hitl_needed      = result.get("hitl_needed", False)
                            context          = result.get("context", [])
                            scores           = result.get("scores", [])
                            sources          = result.get("sources", [])
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
                                     "sources"         : sources,
                                     "_query"          : user_input,
                                 }
                            }
                            st.warning(
                                f"🚨 This query has been escalated for human review.\n\n"
                                f"**Reason:** {hitl_reason}"
                            )
                        else:
                            # Show answer directly (skip if already streamed to screen)
                            _already_streamed = st.session_state.get("stream_mode", True) and not _used_graph_path
                            if not _already_streamed:
                                st.markdown(answer)
                            st.markdown(
                                f'<span style="color:{color}; font-size:0.85em; font-weight:600">'
                                f'{badge}</span>',
                                unsafe_allow_html=True
                            )

                            # Source citations (filename + page)
                            if sources:
                                _cites = []
                                for s in sources:
                                    fn = s.get("source") or s.get("filename") or s.get("collection", "doc")
                                    pg = s.get("page")
                                    _cites.append(f"{fn}" + (f", p.{pg + 1}" if isinstance(pg, int) else ""))
                                _uniq = list(dict.fromkeys(_cites))
                                st.markdown(
                                    '<span style="font-size:0.8em; color:#555">📎 Sources: '
                                    + " · ".join(_uniq) + "</span>",
                                    unsafe_allow_html=True
                                )

                            # Retrieved chunks
                            if context:
                                with st.expander(f"🔍 {len(context)} Retrieved Chunks", expanded=False):
                                    for i, chunk in enumerate(context):
                                        score = scores[i] if i < len(scores) else None
                                        sim   = round((1 - score) * 100, 1) if score is not None else "—"
                                        src   = sources[i] if i < len(sources) else {}
                                        tag   = src.get("source") or src.get("filename") or src.get("collection", "")
                                        if isinstance(src.get("page"), int):
                                            tag += f", p.{src['page'] + 1}"
                                        st.markdown(f"**Chunk {i+1}** | L2 Score: `{round(score,4) if score else '—'}` | Similarity: `{sim}%`" + (f" | 📄 `{tag}`" if tag else ""))
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
                                    "sources"         : sources,
                                    "_query"          : user_input,
                                }
                            })
                            try:
                                db.add_message(
                                    st.session_state["session_id"], "assistant", answer,
                                    confidence=confidence_level,
                                    similarity_pct=similarity_pct,
                                    sources=sources,
                                )
                            except Exception:
                                pass

                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Pipeline error: {e}")
                        st.exception(e)
