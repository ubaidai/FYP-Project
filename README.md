# 🤖 AI Customer Support Agent Builder
### FYP Project — Data Science | NLP | ML | RAG Pipeline

---

## 🧠 What This Project Does
- Upload PDF/TXT knowledge base documents (FAQs, policies, manuals)
- Auto-processes them into a vector database using sentence embeddings
- Customers ask questions → AI answers from your documents (RAG pipeline)
- NLP Sentiment Analysis on every customer message (DistilBERT)
- Analytics dashboard with live charts (Plotly)
- Knowledge Gap Clustering using KMeans ML

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Frontend/UI | Streamlit |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq API (LLaMA 3 - FREE) |
| PDF Processing | PyMuPDF |
| Sentiment Analysis | HuggingFace DistilBERT |
| Clustering | scikit-learn KMeans |
| Charts | Plotly |

---

## 🚀 LOCAL SETUP — STEP BY STEP

### Step 1 — Get FREE Groq API Key
1. Go to https://console.groq.com
2. Sign up (free)
3. Go to API Keys → Create API Key
4. Copy the key (starts with gsk_)

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```
NOTE: This downloads ML models — takes 5-10 minutes. Do it once.

### Step 4 — Add Your API Key
Create a file called .env in the project root and add:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 5 — Run the App
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## ☁️ DEPLOY ON STREAMLIT CLOUD (FREE)

1. Push all files to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect GitHub → select your repo
4. Set Main file: app.py
5. Click Deploy
6. Go to Settings → Secrets → Add:
   GROQ_API_KEY = "gsk_your_key_here"

---

## 📁 Project Structure
```
ai_support_agent/
├── app.py                         ← Main Streamlit app
├── requirements.txt               ← All dependencies
├── .env                           ← Your API key (create this)
├── .env.example                   ← Template
├── .streamlit/
│   └── config.toml                ← Dark theme config
├── utils/
│   ├── __init__.py
│   ├── document_processor.py      ← PDF/TXT extraction & chunking
│   ├── vector_store.py            ← ChromaDB embeddings & retrieval
│   ├── llm_client.py              ← Groq LLM integration
│   └── analytics.py              ← Sentiment analysis & KMeans
└── data/
    └── sample_faq.txt             ← Sample document for testing
```

---

## 🎓 FYP Academic Components
| Component | Concept |
|---|---|
| RAG Pipeline | Information Retrieval + NLP |
| Sentence Embeddings | Deep Learning (Transformer) |
| Vector Similarity Search | ML-based semantic retrieval |
| Sentiment Analysis (DistilBERT) | Fine-tuned NLP Deep Learning |
| KMeans Clustering | Unsupervised Machine Learning |
| Plotly Dashboard | Data Science Visualization |

---

## 🧪 HOW TO DEMO IN FYP PRESENTATION

1. Open app → show 3 tabs
2. Upload data/sample_faq.txt → click Process
3. Go to Chat → ask 5-6 questions → show sentiment badges
4. Ask 2 questions the doc doesn't have (to trigger unanswered tracking)
5. Go to Analytics → show sentiment pie, timeline, conversation log
6. Show Knowledge Gap Clustering at the bottom
7. Explain: "This KMeans ML component clusters unanswered questions so businesses know what to add to their docs"

