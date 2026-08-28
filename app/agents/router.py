from ollama import Client
from app.config import OLLAMA_HOST
from app.agents.retriever import search_documents
from app.agents.generator import generate_response

def route_and_execute(query: str) -> dict:
    """Orchestrator: Decides routing logic and returns final answer."""
    client = Client(host=OLLAMA_HOST, timeout=120.0)

    system_prompt = (
        "You are a router agent. Analyze the user's query and decide if it requires searching "
        "the internal knowledge base (e.g., project details, documentation, company info) "
        "or if it is a general greeting/question.\n"
        "Respond with ONLY one word: 'RETRIEVE' or 'DIRECT'."
    )

    decision = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )["message"]["content"].strip().upper()

    print(f"[Router Decision]: {decision}")

    if "RETRIEVE" in decision:
        docs = search_documents(query)
        answer = generate_response(query, docs)
        return {"mode": "retrieval", "answer": answer, "sources": docs}
    else:
        direct_response = client.chat(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": query}],
        )["message"]["content"]
        return {"mode": "direct", "answer": direct_response, "sources": []}

if __name__ == "__main__":
    print("--- Test 1: Knowledge Query ---")
    res1 = route_and_execute("What database and search technologies are used in Project Alpha?")
    print(f"Answer: {res1['answer']}\n")

    print("--- Test 2: General Greeting ---")
    res2 = route_and_execute("Hello, how are you?")
    print(f"Answer: {res2['answer']}")
