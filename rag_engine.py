import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from groq import Groq

# ── Embedding model (runs locally) ────────────────────────────────────────
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── ChromaDB (in-memory) ───────────────────────────────────────────────────
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))


def _get_or_create_collection(bot_id: str):
    name = f"bot_{re.sub(r'[^a-zA-Z0-9_-]', '_', bot_id)}"
    return chroma_client.get_or_create_collection(name)


# ── Document ingestion ─────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_document(file_bytes: bytes, filename: str, bot_id: str) -> int:
    """Process and store document chunks. Returns number of chunks stored."""
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = EMBED_MODEL.encode(chunks).tolist()
    collection = _get_or_create_collection(bot_id)

    # Remove old chunks for this file to avoid duplicates on re-upload
    existing = collection.get()
    if existing["ids"]:
        ids_to_delete = [
            eid for eid, meta in zip(existing["ids"], existing["metadatas"])
            if meta.get("filename") == filename
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"filename": filename} for _ in chunks],
    )
    return len(chunks)


# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieve_context(query: str, bot_id: str, top_k: int = 5):
    """Returns (context_string, confidence_score 0-1)."""
    collection = _get_or_create_collection(bot_id)
    if collection.count() == 0:
        return "", 0.0

    query_embedding = EMBED_MODEL.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    if not results["documents"] or not results["documents"][0]:
        return "", 0.0

    docs = results["documents"][0]
    distances = results["distances"][0]

    # Convert L2 distance to 0-1 confidence score
    confidence = float(max(0.0, 1.0 - (min(distances) / 2.0)))
    context = "\n\n---\n\n".join(docs)
    return context, confidence


# ── LLM Answer Generation ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful customer support agent. Answer the user's question
using ONLY the context provided below. If the answer is not in the context, say:
"I don't have information about that in my knowledge base. Please contact our support team directly."
Be concise, friendly, and professional. Do not make up information."""


def generate_answer(
    query: str,
    context: str,
    chat_history: list,
    confidence: float,
    groq_api_key: str,
) -> str:
    if confidence < 0.25 or not context:
        return (
            "I don't have enough information to answer that accurately. "
            "Please contact our support team directly for assistance."
        )

    client = Groq(api_key=groq_api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in chat_history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content


def get_collection_count(bot_id: str) -> int:
    return _get_or_create_collection(bot_id).count()
