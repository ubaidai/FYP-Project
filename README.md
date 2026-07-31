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

## Evaluation

Retrieval is measured against a labelled dataset of 55 questions — 45 in-scope across 15 topics, phrased as a customer would ask rather than copying the FAQ's wording, plus 10 deliberately unanswerable ones.

```bash
python eval/run_eval.py --sweep
```

| chunk_size | chunks | hit@1 | MRR | in-scope vs off-topic separation |
|---:|---:|---:|---:|---:|
| **400** (shipped default) | 2 | 66.7% | 0.833 | +0.145 |
| 200 | 3 | 77.8% | 0.870 | +0.151 |
| 40 | 15 | 62.2% | 0.764 | +0.220 |

The shipped default is the weakest configuration on hit@1, and separation between answerable and unanswerable questions improves monotonically as chunks shrink. The corpus is small enough that the hit-rate differences sit inside the noise band for 45 queries, so the chunking result is directional rather than proven — [eval/README.md](eval/README.md) sets out the full table and what it does and doesn't support.

## Known limitations

Being explicit about these, because they're the difference between a demo and a production system:

- **The vector store is in-memory and session-scoped.** Embeddings live in `st.session_state` and are lost on reload. Fine for a single-user demo; a persistent store (pgvector, Qdrant, Chroma) is required for real deployment. `get_chroma_client()` is a stub returning `None` — kept as the seam where a real client would be injected.
- **Intent classification is regex/keyword-based**, not a trained model. It's deterministic and fast, but brittle on phrasing it hasn't seen. A fine-tuned classifier is the upgrade path.
- **The default chunk size makes retrieval nearly a no-op on the sample corpus.** At `chunk_size=400`, the 442-word sample FAQ becomes 2 chunks, so retrieving the top 4 returns everything. Measured hit@1 is 66.7% — the worst of six configurations tested. See [eval/](eval/).
- **No abstention gate.** The app sends the top 4 chunks to the LLM regardless of similarity score and infers "unanswered" by string-matching the reply. Measured ceiling for a similarity threshold is 87.3% accept/reject accuracy.
- **Answer quality is unmeasured.** The harness scores retrieval only; there is no LLM-judge pass over generated answers yet.
- **Two dead modules ship in the repo.** `rag_engine.py` at the root is a complete alternative implementation backed by ChromaDB, and `utils/rag_engine.py` is a third variant. Neither is imported by `app.py`, and `chromadb` is absent from `requirements.txt`, so the root module would fail on a clean install. Removal or consolidation pending.

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

- [x] Labelled evaluation set and retrieval harness — hit-rate@k, MRR, score separation
- [ ] Larger corpus so chunking differences clear the noise floor
- [ ] Confidence gating via `retrieve_with_scores`, with human handoff below threshold
- [ ] LLM-judge pass to score answer quality, not just retrieval
- [ ] Persistent vector store (pgvector or Qdrant)
- [ ] Replace rule-based intent classifier with a fine-tuned model
- [ ] Remove the two dead `rag_engine` modules

---

Built by [Ubaid Ur Rehman](https://github.com/ubaidai) · [ubaidai.com](https://ubaidai.com)
