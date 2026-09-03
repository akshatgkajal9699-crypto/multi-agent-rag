import json
import ollama
import app.config as config

OLLAMA_HOST = getattr(config, 'OLLAMA_HOST', getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
LLM_MODEL = getattr(config, 'LLM_MODEL', getattr(config, 'MODEL_NAME', 'llama3'))

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def route_and_execute(query: str):
    prompt = f"""You are a router agent. Determine if this query requires document retrieval or if it is general knowledge.
Query: "{query}"

Respond strictly with valid JSON using this format:
{{"mode": "retrieve", "action": "retrieve", "query": "{query}"}}
OR
{{"mode": "direct", "action": "direct", "query": "{query}"}}"""

    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip()
        data = json.loads(content)
        
        # Ensure 'mode' key exists regardless of LLM variations
        if "mode" not in data:
            data["mode"] = data.get("action", "retrieve")
        return data
    except Exception:
        return {"mode": "retrieve", "action": "retrieve", "query": query}

def run_rag_pipeline(query: str):
    return route_and_execute(query)
