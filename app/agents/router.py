import ollama
from app.config import OLLAMA_HOST

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def route_query(query: str) -> str:
    prompt = f"""
    You are a routing agent. Decide if the user's query requires retrieving documents from a database or if it can be answered directly.
    Respond with ONLY 'RETRIEVE' or 'DIRECT'.
    
    Query: {query}
    """
    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    decision = response["message"]["content"].strip().upper()
    return "RETRIEVE" if "RETRIEVE" in decision else "DIRECT"

# Alias to satisfy main.py import
route_and_execute = route_query
