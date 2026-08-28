import ollama
from app.config import OLLAMA_HOST
from app.agents.hybrid_retriever import hybrid_search

client = ollama.Client(host=OLLAMA_HOST)

def generate_rag_response(query: str, docs: list = None) -> dict:
    if docs is None:
        docs = hybrid_search(query, top_k=3)
    
    if not docs:
        context_str = "No relevant documents found."
        sources = []
    else:
        context_str = "\n\n".join([d["content"] for d in docs])
        sources = docs

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question concisely."
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

generate_response = generate_rag_response
