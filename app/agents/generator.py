import ollama
from app.config import OLLAMA_HOST
from app.agents.hybrid_retriever import hybrid_search

client = ollama.Client(host=OLLAMA_HOST)

def generate_rag_response(query: str, docs: list = None) -> dict:
    """Generates a grounded response using provided docs or performs hybrid retrieval if missing."""
    if docs is None:
        docs = hybrid_search(query, top_k=3)
    
    if not docs:
        context_str = "No relevant documents found."
        sources = []
    else:
        context_str = "\n\n".join([d["content"] if isinstance(d, dict) else d[0] for d in docs])
        sources = docs

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you do not know the answer, say that you don't know. "
        "Keep the response concise and accurate."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources
    }

# Function alias for compatibility with router imports
generate_response = generate_rag_response
