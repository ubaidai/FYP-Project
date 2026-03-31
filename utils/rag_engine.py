import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load embedding model once
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class RAGEngine:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection_name = f"support_kb_{uuid.uuid4().hex[:8]}"
        self.collection = self.client.create_collection(self.collection_name)
        self.embedder = get_embedder()

    def build_index(self, chunks: list[str]):
        """Embed chunks and store in ChromaDB."""
        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

    def retrieve(self, query: str, top_k: int = 4):
        """Retrieve top_k relevant chunks for the query."""
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = 1 - (distances[0] / 2) if distances else 0.0
        confidence = max(0.0, min(1.0, confidence))
        return docs, confidence

    def query(self, user_question: str, bot_name: str = "Aria",
              persona: str = "Professional & Formal", company: str = "") -> tuple[str, float]:
        """Full RAG pipeline: retrieve + generate."""
        # Retrieve relevant chunks
        context_chunks, confidence = self.retrieve(user_question)
        context = "\n\n".join(context_chunks)

        # Build persona instruction
        persona_map = {
            "Professional & Formal": "You are professional, formal, and precise.",
            "Friendly & Casual": "You are friendly, warm, and conversational.",
            "Concise & Direct": "You are concise and direct. Keep answers short.",
            "Empathetic & Warm": "You are empathetic, patient, and reassuring."
        }
        persona_instruction = persona_map.get(persona, persona_map["Professional & Formal"])

        company_context = f"for {company}" if company else ""

        system_prompt = f"""You are {bot_name}, an AI customer support agent {company_context}.
{persona_instruction}

Your job is to answer customer questions ONLY based on the provided context below.
If the answer is not in the context, say: "I don't have information about that in my knowledge base. Please contact our support team for further help."
Never make up information. Be helpful and accurate.

CONTEXT:
{context}
"""

        # Call Groq API
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=512,
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
        return answer, confidence
