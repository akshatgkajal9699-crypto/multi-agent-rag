import json
import ollama
import app.config as config
from app.agents.retriever import get_embedding
from app.agents.generator import generate_answer

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

    mode = "retrieve"
    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip()
        data = json.loads(content)
        mode = data.get("mode", data.get("action", "retrieve"))
    except Exception:
        mode = "retrieve"

    # Execute downstream workflow based on mode
    if mode == "direct":
        answer = generate_answer(query)
    else:
        # Generate embedding and answer with context
        try:
            _ = get_embedding(query)
        except Exception:
            pass
        answer = generate_answer(query)

    return {
        "mode": mode,
        "action": mode,
        "answer": answer,
        "response": answer,
        "query": query
    }

def run_rag_pipeline(query: str):
    return route_and_execute(query)
