import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List
import streamlit as st


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(anonymized_telemetry=False))


def create_or_get_collection(client, collection_name: str = "support_docs"):
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )


def add_chunks_to_vectorstore(chunks: List[str], collection, model) -> int:
    if not chunks:
        return 0
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"chunk_{i}_{abs(hash(chunk)) % 100000}" for i, chunk in enumerate(chunks)]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return len(chunks)


def retrieve_relevant_chunks(query: str, collection, model, n_results: int = 4) -> List[str]:
    query_embedding = model.encode([query]).tolist()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, count)
    )
    return results['documents'][0] if results['documents'] else []
