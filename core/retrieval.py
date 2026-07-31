"""Streamlit-free retrieval core.

Both the Streamlit app (via utils.vector_store) and the offline evaluation
harness import from here, so the numbers the harness reports describe the
same code path the app actually runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50
DEFAULT_TOP_K = 4

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def load_model(name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load and cache an embedding model by name."""
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split text into overlapping word windows.

    Mirrors utils.document_processor.chunk_text so evaluation results apply
    to the shipped ingestion path.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


@dataclass
class VectorIndex:
    """In-memory vector index using cosine similarity.

    Embeddings are L2-normalised on insert so similarity is a plain dot
    product and scores fall in [-1, 1].
    """

    model: SentenceTransformer
    documents: List[str] = field(default_factory=list)
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def add(self, chunks: List[str]) -> int:
        if not chunks:
            return 0
        vectors = self.model.encode(chunks, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10

        self.documents.extend(chunks)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )
        return len(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """Return up to `top_k` (chunk, cosine_score) pairs, best first."""
        if self._matrix is None or not self.documents:
            return []

        q = np.asarray(self.model.encode([query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-10

        scores = self._matrix @ q
        k = min(top_k, len(self.documents))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.documents[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.documents)


def build_index(
    text: str,
    model: SentenceTransformer | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> VectorIndex:
    """Chunk `text`, embed it, and return a ready-to-query index."""
    index = VectorIndex(model=model or load_model())
    index.add(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    return index
