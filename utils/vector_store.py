import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import streamlit as st


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_chroma_client():
    return None


def create_or_get_collection(client, collection_name: str = "support_docs"):
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = {"documents": [], "embeddings": []}
    return st.session_state.vector_store


def add_chunks_to_vectorstore(chunks: List[str], collection, model) -> int:
    if not chunks:
        return 0
    embeddings = model.encode(chunks, show_progress_bar=False)
    collection["documents"].extend(chunks)
    collection["embeddings"].extend(embeddings.tolist())
    return len(chunks)


def retrieve_relevant_chunks(query: str, collection, model, n_results: int = 4) -> List[str]:
    if not collection["documents"]:
        return []
    query_embedding = model.encode([query])[0]
    embeddings = np.array(collection["embeddings"])
    query_vec = np.array(query_embedding)
    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [collection["documents"][i] for i in top_indices]
