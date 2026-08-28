import os
from ollama import Client
from app.config import OLLAMA_HOST

def generate_response(query: str, contexts: list[dict]) -> str:
    """Generates an answer grounded strictly in retrieved context chunks."""
    client = Client(host=OLLAMA_HOST, timeout=120.0)

    formatted_context = ""
    for idx, doc in enumerate(contexts, start=1):
        formatted_context += f"[Chunk {idx}] ({doc['metadata'].get('source', 'Unknown')}):\n{doc['content']}\n\n"

    system_prompt = (
        "You are an assistant designed to answer user questions accurately based ONLY on the provided context.\n"
        "Rules:\n"
        "1. Ground your answer strictly in the context provided.\n"
        "2. Add inline citations such as [Chunk 1] or [Chunk 2] whenever citing facts.\n"
        "3. If the answer cannot be found in the context, state clearly: 'I do not have enough information to answer this question based on the provided documents.'"
    )

    user_prompt = f"Contexts:\n{formatted_context}\n\nQuestion: {query}"

    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"]

if __name__ == "__main__":
    from app.agents.retriever import search_documents
    test_query = "What database and search technologies are used in Project Alpha?"
    retrieved_docs = search_documents(test_query)
    answer = generate_response(test_query, retrieved_docs)
    print("\n--- ANSWER ---")
    print(answer)
