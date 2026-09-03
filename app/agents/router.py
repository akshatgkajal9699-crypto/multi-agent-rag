import json
import ollama
import app.config as config

# Safely extract host and model, falling back to defaults if missing in config
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
{{"action": "retrieve", "query": "{query}"}}
OR
{{"action": "direct", "query": "{query}"}}"""

    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip()
        return json.loads(content)
    except Exception:
        return {"action": "retrieve", "query": query}

def run_rag_pipeline(query: str):
    return route_and_execute(query)
