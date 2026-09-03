import json
import ollama
from app.config import OLLAMA_HOST, LLM_MODEL

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def route_and_execute(query: str):
    prompt = f"""You are a router agent. Determine if this query requires document retrieval or if it is general knowledge.
Query: "{query}"

Respond strictly with valid JSON using this format:
{{"action": "retrieve", "query": "{query}"}}
OR
{{"action": "direct", "query": "{query}"}}"""

    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response['message']['content'].strip()
    try:
        return json.loads(content)
    except Exception:
        return {"action": "retrieve", "query": query}
