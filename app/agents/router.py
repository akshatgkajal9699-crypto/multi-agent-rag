import ollama
from app.config import OLLAMA_HOST
from app.agents.hybrid_retriever import hybrid_search
from app.agents.generator import generate_response

client = ollama.Client(host=OLLAMA_HOST)

ROUTER_PROMPT = """
You are an intent router for a RAG system.
Classify the user's query into one of two categories:
1. 'RETRIEVE': The query requires specific domain knowledge, factual lookup, or technical documentation.
2. 'DIRECT': The query is a general greeting, conversational statement, or simple question requiring no documentation.

Respond with ONLY one word: either 'RETRIEVE' or 'DIRECT'.
"""

def route_and_execute(query: str) -> dict:
    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query}
        ],
        options={"temperature": 0.0}
    )

    decision = response["message"]["content"].strip().upper()
    print(f"[Router Decision]: {decision}")

    if "RETRIEVE" in decision:
        docs = hybrid_search(query, top_k=3)
        gen_result = generate_response(query, docs)
        return {
            "mode": "retrieval",
            "answer": gen_result["answer"],
            "sources": gen_result["sources"]
        }
    else:
        direct_response = client.chat(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": query}]
        )
        return {
            "mode": "direct",
            "answer": direct_response["message"]["content"],
            "sources": []
        }
