# RAG Customer Support Assistant

A production-ready RAG system with dynamic PDF ingestion, LangGraph routing, HITL support, and a Streamlit chat UI.

## Stack
| Component   | Tech                             |
|-------------|----------------------------------|
| Embeddings  | Gemini `gemini-embedding-001`    |
| LLM         | Groq `llama-3.3-70b` or Gemini  |
| Vector DB   | ChromaDB (persistent)            |
| Orchestration | LangGraph                      |
| UI          | Streamlit                        |

## Setup
```bash
# 1. Clone and activate venv
python -m venv venv && venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and GROQ_API_KEY

# 4. Run
streamlit run app/streamlit_app.py
```

## Key Features

### Duplicate Ingestion Prevention
Each uploaded file is MD5-hashed. If the same file is uploaded again, the previously built ChromaDB collection is reused — no re-embedding occurs.

### Session-Isolated Collections
Each PDF gets its own persistent ChromaDB collection named `doc_<md5hash>`. Collections persist across restarts and can be switched or deleted from the sidebar.

### Evaluation & HITL
- Threshold: `avg_score < 0.4` → confident
- If low confidence → HITL panel appears in UI
- Human can override the AI answer or accept it

### LLM Provider Switch
Set `LLM_PROVIDER=groq` or `LLM_PROVIDER=gemini` in `.env`. Generation model switches automatically; embeddings always use Gemini.

## Project Structure
```
app/
  config.py            # All env config
  streamlit_app.py     # Streamlit UI
ingestion/
  load_pdf.py          # PDF loader
  chunking.py          # Text cleaner + splitter
  embedding.py         # Gemini embeddings w/ rate limiter
  vector_store.py      # ChromaDB helpers
retrieval/
  retriever.py         # Similarity search w/ scores
llm/
  prompt_template.py   # Prompt builder
  gemini_model.py      # Groq / Gemini LLM calls
graph/
  state.py             # LangGraph state
  nodes.py             # Retrieval, generation, evaluation, HITL nodes
  workflow.py          # Graph compile + run
hitl/
  escalation.py        # CLI HITL (legacy)
```

## Sample Queries
After uploading a product/service manual or policy PDF:
- "What is the return policy?"
- "How do I reset my password?"
- "What are the supported payment methods?"
- "Explain the subscription cancellation process."
- "What warranty is provided for hardware products?"
