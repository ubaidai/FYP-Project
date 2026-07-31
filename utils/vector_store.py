"""Streamlit-facing wrapper around the retrieval core.

The actual embedding and similarity search live in `core.retrieval`, which has
no Streamlit dependency, so the offline evaluation harness in `eval/` exercises
exactly the code path this app runs. This module only adds Streamlit caching
and session-scoped storage on top.
"""

from typing import List, Tuple

import streamlit as st

from core.retrieval import DEFAULT_MODEL, DEFAULT_TOP_K, VectorIndex, load_model


@st.cache_resource
def load_embedding_model():
    return load_model(DEFAULT_MODEL)


def get_chroma_client():
    """Retained for API compatibility.

    The index is in-memory and session-scoped, so there is no external client.
    This is the seam where a persistent store (pgvector, Qdrant, Chroma) would
    be injected — see the roadmap in the README.
    """
    return None


def create_or_get_collection(client, collection_name: str = "support_docs"):
    """Return the session's VectorIndex, creating it on first use."""
    key = f"vector_index::{collection_name}"
    if key not in st.session_state:
        st.session_state[key] = VectorIndex(model=load_embedding_model())
    return st.session_state[key]


def add_chunks_to_vectorstore(chunks: List[str], collection, model=None) -> int:
    """Embed and store chunks. Returns the number added."""
    return collection.add(chunks)


def retrieve_relevant_chunks(
    query: str, collection, model=None, n_results: int = DEFAULT_TOP_K
) -> List[str]:
    """Top-n most similar chunks, best first."""
    return [chunk for chunk, _score in collection.search(query, top_k=n_results)]


def retrieve_with_scores(
    query: str, collection, n_results: int = DEFAULT_TOP_K
) -> List[Tuple[str, float]]:
    """Top-n chunks with cosine scores, for confidence gating.

    See eval/results.md for the measured threshold that separates answerable
    questions from off-topic ones on the sample knowledge base.
    """
    return collection.search(query, top_k=n_results)
