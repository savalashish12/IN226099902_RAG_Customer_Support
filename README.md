# 🏢 TechNova Internal Support Assistant

> **RAG-Based Enterprise Customer Support System**  
> Powered by Gemini Embeddings · Groq Llama 3.3 70B · ChromaDB · LangGraph · Streamlit

---

## Overview

TechNova Internal Support Assistant is a **production-grade AI assistant** that answers employee questions from company PDF documents (HR Policy, IT Helpdesk Manual) using Retrieval-Augmented Generation (RAG). It retrieves the most relevant policy chunks, generates grounded answers, measures its own confidence, and escalates low-confidence or complex queries to a **human agent via a real LangGraph HITL (Human-in-the-Loop)** mechanism.

---

## Key Features

| Feature | Detail |
|---|---|
| **PDF Upload with MD5 Dedup** | Same file never re-embedded — reuses existing ChromaDB collection |
| **Session-Isolated Collections** | Each PDF gets its own `doc_<md5>` collection; switch between them from sidebar |
| **MMR Retrieval** | Maximal Marginal Relevance search (k=3 from 15 candidates) for diverse, non-redundant context |
| **No-Answer Guard** | If chunks are too dissimilar (avg L2 ≥ 0.55), LLM is not called — returns "not found" |
| **Confidence Meter** | 🟢 HIGH (<30% L2) · 🟡 MEDIUM (30–55%) · 🔴 LOW (≥55%) with similarity % badge |
| **Retrieved Chunk Viewer** | Expandable panel showing chunk text, L2 score, and similarity % |
| **Real LangGraph HITL** | Uses `interrupt()` + `MemorySaver` + `Command(resume=...)` — true graph pause/resume |
| **Escalation Log** | Sidebar shows last 5 escalations with PENDING / APPROVED / OVERRIDDEN status |
| **Collection Manager** | Switch active KB or delete collections from the sidebar |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Embedding** | Google Gemini `models/gemini-embedding-001` (768-dim, task-typed) |
| **LLM** | Groq `llama-3.3-70b-versatile` (temp=0.1, max_tokens=1024) |
| **Vector DB** | ChromaDB (persistent, L2 distance metric) |
| **Orchestration** | LangGraph `StateGraph` + `MemorySaver` checkpointer |
| **UI** | Streamlit |
| **PDF Loading** | LangChain `PyPDFLoader` |
| **Chunking** | `RecursiveCharacterTextSplitter` (size=400, overlap=80) |
| **Retry / Rate Limit** | `tenacity` (exponential backoff, 6 attempts) |

---

## Architecture

```
PDF Upload ──▶ MD5 Dedup ──▶ PyPDFLoader ──▶ Noise Cleanup
    ──▶ Chunk (400/80) ──▶ Gemini Embed ──▶ ChromaDB (doc_<md5>)

Query ──▶ LangGraph Pipeline:
  [retrieval_node]  MMR search → context + L2 scores
  [generation_node] No-Answer Guard → Groq LLM → answer
  [evaluation_node] Confidence: HIGH | MEDIUM | LOW
        │
        ├── HIGH/MEDIUM (simple) ──▶ Direct answer to UI
        └── LOW / urgent / complex ──▶ [hitl_node]
                                         interrupt() → PAUSED
                                         Human: Approve / Override
                                         Command(resume) → RESUMED
                                         Final answer → UI
```

---

## Project Structure

```
Rag-Customer-Support/
│
├── app/
│   ├── config.py              # All env-based configuration
│   └── streamlit_app.py       # UI: upload, chat, HITL panel, escalation log
│
├── ingestion/
│   ├── load_pdf.py            # PDF → chunk → embed → store orchestrator
│   ├── chunking.py            # Noise regex + RecursiveCharacterTextSplitter
│   ├── embedding.py           # RateLimitedEmbeddings (Gemini + tenacity)
│   └── vector_store.py        # ChromaDB CRUD: store, exists, list, delete
│
├── retrieval/
│   └── retriever.py           # MMR search + L2 score mapping + collection cache
│
├── llm/
│   ├── groq_client.py         # Groq API call, No-Answer Guard
│   └── prompt_template.py     # XML-tagged prompt builder
│
├── graph/
│   ├── state.py               # RAGState TypedDict (14 fields)
│   ├── nodes.py               # 4 nodes: retrieval, generation, evaluation, hitl
│   └── workflow.py            # StateGraph compile, run_rag_pipeline(), resume()
│
├── hitl/
│   └── manager.py             # HITLManager: session lifecycle + audit log
│
├── chroma_db/                 # Persistent ChromaDB storage (auto-created)
├── HLD.md                     # High-Level Design document
├── LLD.md                     # Low-Level Design document
├── TECH_DOC.md                # Full technical documentation
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key (free tier works)
- Groq API key (free tier works)

### Installation

```bash
# 1. Clone repo and create virtual environment
git clone <repo-url>
cd Rag-Customer-Support
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set your API keys:
#   GEMINI_API_KEY=your_gemini_key
#   GROQ_API_KEY=your_groq_key

# 5. Run the application
streamlit run app/streamlit_app.py
```

### Environment Variables (`.env`)

```env
GEMINI_API_KEY=your_google_ai_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_BATCH_SIZE=25
EMBEDDING_DELAY_SECONDS=5

HIGH_CONF_THRESHOLD=0.30
MEDIUM_CONF_THRESHOLD=0.55
TOP_K=5
```

---

## How to Use

1. **Upload a PDF** from the left sidebar (HR Policy, IT Manual, or any company document)
2. The system chunks, embeds, and stores it in ChromaDB — takes ~30-60s for a typical PDF
3. **Ask a question** in the chat box at the bottom
4. The system retrieves relevant chunks, generates an answer, and shows a confidence badge
5. If confidence is low or the query is complex/urgent, a **HITL review panel** appears
6. The human agent can **Approve** the AI answer or **Override** it with a correct answer
7. All escalations are logged in the sidebar **Escalation Log**

---

## Confidence Thresholds

| L2 Distance Range | Confidence | HITL Triggered? |
|---|---|---|
| < 0.30 | 🟢 HIGH | Never |
| 0.30 – 0.55 | 🟡 MEDIUM | Only if urgent keywords or query > 10 words |
| ≥ 0.55 | 🔴 LOW | Always |
| Any (not-found phrase in answer) | 🔴 LOW | Always |

**Urgent keywords that trigger HITL at MEDIUM confidence:**  
`urgent`, `critical`, `emergency`, `not working`, `failed`, `error`, `broken`, `crash`, `data loss`, `security breach`

---

## Sample Queries

### HR Policy
- `"How many days of annual leave do employees get?"`
- `"What is the resignation notice period?"`
- `"When is salary credited each month?"`
- `"What are the PF contribution rules?"`

### IT Helpdesk
- `"How do I reset my VPN password?"`
- `"What is the SLA for a Priority 1 ticket?"`
- `"How do I request software installation?"`
- `"What are the steps for laptop imaging?"`

### HITL-Triggering Queries
- `"My laptop crashed and I lost all my project data — this is critical"` → urgent keyword
- `"Can you explain the complete onboarding process with all required HR forms and IT setup steps?"` → complex (>10 words)
- `"What is the company stock price?"` → out-of-scope

---

## Documentation

| File | Description |
|---|---|
| [`HLD.md`](./HLD.md) | High-Level Design: architecture, components, data flow, technology choices, scalability |
| [`LLD.md`](./LLD.md) | Low-Level Design: module design, data structures, node logic, routing tables, HITL flow, error handling |
| [`TECH_DOC.md`](./TECH_DOC.md) | Full Technical Documentation: RAG explanation, design decisions, workflow, challenges, testing strategy, future enhancements |

---

## Design Principles

- **Grounded generation:** LLM answers only from retrieved context — hallucination is structurally prevented
- **Transparent confidence:** Every answer shows its L2-based similarity score
- **Fail-safe escalation:** When uncertain, the system escalates rather than guessing
- **MD5 deduplication:** Same document is never embedded twice
- **Real HITL:** Uses LangGraph's native `interrupt()`/`resume()` — not a simulated workaround
- **Stateless queries:** Each query is isolated by `thread_id`; no cross-session leakage

---

*TechNova Internal Support Assistant · Stack: Gemini + Groq + ChromaDB + LangGraph + Streamlit*
