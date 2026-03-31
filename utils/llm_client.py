from groq import Groq
from typing import List, Dict


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def generate_response(
    client: Groq,
    user_message: str,
    context_chunks: List[str],
    chat_history: List[Dict],
    bot_name: str = "Support Agent",
    company_name: str = "Our Company"
) -> str:
    context = "\n\n".join(context_chunks) if context_chunks else "No specific context found."

    system_prompt = f"""You are {bot_name}, an intelligent customer support assistant for {company_name}.

Answer customer questions based ONLY on the knowledge base context below.

KNOWLEDGE BASE:
{context}

RULES:
- Answer ONLY from the context above
- If the answer is not in context, say: "I don't have information about that. Please contact our support team directly."
- Be friendly, concise, and professional
- Never make up information
- If the question is a greeting, respond warmly and ask how you can help
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=512,
        temperature=0.3
    )
    return response.choices[0].message.content
