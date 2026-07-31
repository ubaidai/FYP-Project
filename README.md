# AI Customer Support Agent

A retrieval-augmented support agent that answers customer questions from your own documents, classifies intent and sentiment on every message, and surfaces the questions your documentation fails to answer.

Upload your FAQs, policies or manuals as PDF/TXT. The app chunks and embeds them, answers incoming questions from that corpus, and builds an analytics view showing what customers actually ask — including the gaps.

---

## Why this exists

Most support chatbots answer questions. The harder problem is knowing **which questions you can't answer**. This project tracks unanswered queries and clusters them with KMeans, so a support team gets a ranked list of documentation gaps rather than a transcript to read through.

---

## Features

| Feature | Implementation |
|---|---|
| Document ingestion | PDF/TXT parsing via PyMuPDF, chunked for retrieval |
| Semantic retrieval | `all-MiniLM-L6-v2` sentence embeddings, cosine similarity top-k |
| Answer generation | Groq API (LLaMA 3), answers grounded in retrieved chunks |
| Sentiment analysis | HuggingFace DistilBERT on every customer message |
| Intent routing | Rule-based classifier — billing / technical / returns / general |
| Knowledge-gap analysis | KMeans clustering over unanswered questions |
| Analytics dashboard | Plotly — sentiment distribution, volume over time, conversation log |

---

## Architecture

```
PDF / TXT upload
      │
      ▼
 PyMuPDF extract ──► chunk ──► all-MiniLM-L6-v2 embed ──► in-memory vector store
                                                                    │
 customer question ──► embed ──► cosine similarity top-k ───────────┘
      │                                                    │
      │                                                    ▼
      │                                         retrieved context
      │                                                    │
      ├──► DistilBERT sentiment                            ▼
      ├──► rule-based intent                    Groq / LLaMA 3 answer
      │                                                    │
      └──► unanswered? ──► KMeans gap clustering ◄─────────┘
                                    │
                                    ▼
                          Plotly analytics dashboard
```

---

## Tech stack

**Python 3.11** · Streamlit · sentence-transformers · HuggingFace Transformers (DistilBERT) · Groq API (LLaMA 3) · PyMuPDF · scikit-learn · NumPy · Plotly · pandas

---

## Known limitations

Being explicit about these, because they're the difference between a demo and a production system:

- **The vector store is in-memory and session-scoped.** Embeddings live in `st.session_state` and are lost on reload. Fine for a single-user demo; a persistent store (pgvector, Qdrant, Chroma) is required for real deployment. `get_chroma_client()` is a stub returning `None` — kept as the seam where a real client would be injected.
- **Intent classification is regex/keyword-based**, not a trained model. It's deterministic and fast, but brittle on phrasing it hasn't seen. A fine-tuned classifier is the upgrade path.
- **No retrieval evaluation yet.** There is no labelled test set and no hit-rate or answer-accuracy measurement, so retrieval quality is currently unverified. This is the next thing being added.
- **Chunking is fixed-size**, with no semantic or overlap-aware strategy tuned against a benchmark.
- `rag_engine.py` exists at both the repo root and in `utils/` — consolidation pending.

---

## Setup

**1. Get a Groq API key** (free) at [console.groq.com](https://console.groq.com) → API Keys → Create.

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

First run downloads the embedding and sentiment models — expect 5–10 minutes.

**4. Add your key** — create `.env` in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

**5. Run**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. A sample knowledge base is included at `data/sample_faq.txt`.

---

## Deploying

Push to GitHub, connect the repo at [share.streamlit.io](https://share.streamlit.io), set the main file to `app.py`, and add `GROQ_API_KEY` under **Settings → Secrets**.

---

## Project structure

```
├── app.py                        Streamlit UI and application flow
├── rag_engine.py                 Retrieval + generation orchestration
├── sentiment_analyzer.py         DistilBERT sentiment inference
├── requirements.txt
├── runtime.txt                   Python 3.11
├── data/
│   └── sample_faq.txt            Sample knowledge base for testing
└── utils/
    ├── document_processor.py     PDF/TXT extraction and chunking
    ├── vector_store.py           Embedding storage and similarity search
    ├── llm_client.py             Groq API client
    ├── rag_engine.py             Retrieval helpers
    ├── intent_classifier.py      Rule-based intent routing
    └── analytics.py              Sentiment aggregation and KMeans clustering
```

---

## Roadmap

- [ ] Labelled evaluation set — measure retrieval hit-rate@k and answer accuracy
- [ ] Persistent vector store (pgvector or Qdrant)
- [ ] Semantic chunking with overlap, benchmarked against fixed-size
- [ ] Replace rule-based intent classifier with a fine-tuned model
- [ ] Confidence gating with human handoff on low-confidence answers

---

Built by [Ubaid Ur Rehman](https://github.com/ubaidai) · [ubaidai.com](https://ubaidai.com)
