"""
Analytics — Admin dashboard for the TechNova Support Assistant.

Surfaces operational metrics from the SQLite store:
  - KPI cards: queries, escalation rate, confidence mix, feedback score
  - Charts: query volume over time, confidence distribution
  - Knowledge-gap report: low-confidence & negatively-rated questions
  - Document usage: which sources are cited most
  - Escalation audit log

Access is gated by an optional APP_PASSWORD (see app/streamlit_app.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from db import database as db

st.set_page_config(page_title="Analytics · TechNova", page_icon="📊", layout="wide")

db.init_db()

st.markdown("# 📊 Support Analytics")
st.caption("Operational insight into the RAG assistant's usage, quality, and knowledge gaps.")

stats = db.get_stats()
conf  = stats["confidence_dist"]
fb    = stats["feedback_dist"]
total_q = stats["total_queries"]
up   = fb.get("up", 0)
down = fb.get("down", 0)
fb_score = round(100 * up / (up + down), 1) if (up + down) else 0.0

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Queries", f"{total_q}")
c2.metric("Escalation Rate", f"{round(db.get_escalation_rate() * 100, 1)}%",
          help="Share of queries routed to a human via HITL")
c3.metric("Positive Feedback", f"{fb_score}%", help="thumbs-up / (up + down)")
_high = conf.get("HIGH", 0)
_denom = sum(conf.values()) or 1
c4.metric("High-Confidence Rate", f"{round(100 * _high / _denom, 1)}%")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────────────────────
ch1, ch2 = st.columns(2)

with ch1:
    st.subheader("Queries (last 14 days)")
    qpd = db.get_queries_per_day(14)
    if qpd:
        st.bar_chart(qpd, height=280)
    else:
        st.caption("No query data yet.")

with ch2:
    st.subheader("Confidence Distribution")
    if conf:
        st.bar_chart(conf, height=280)
    else:
        st.caption("No answered queries yet.")

st.markdown("---")

# ── Knowledge gaps ───────────────────────────────────────────────────────────
st.subheader("Knowledge Gap Report")
st.caption("Low-confidence answers and negatively-rated questions - candidate topics to document.")
gaps = db.get_knowledge_gaps(30)
if gaps:
    st.dataframe(gaps, use_container_width=True, hide_index=True)
else:
    st.success("No knowledge gaps detected yet.")

st.markdown("---")

# ── Document usage ───────────────────────────────────────────────────────────
st.subheader("Document Usage (by citations)")
usage = db.get_document_usage(20)
if usage:
    st.dataframe(usage, use_container_width=True, hide_index=True)
else:
    st.caption("No answers with citations yet.")

st.markdown("---")

# ── Escalation audit log ─────────────────────────────────────────────────────
st.subheader("Escalation Audit Log")
esc = db.get_escalations(100)
if esc:
    st.dataframe(
        [
            {
                "query": e["query"],
                "status": e["status"],
                "confidence": e["confidence"],
                "reason": e["reason"],
                "created": (e["created_at"] or "")[:19],
                "resolved": (e["resolved_at"] or "")[:19],
            }
            for e in esc
        ],
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No escalations recorded.")
