import ollama
import app.config as config

OLLAMA_HOST = getattr(config, 'OLLAMA_HOST', getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
LLM_MODEL = getattr(config, 'LLM_MODEL', getattr(config, 'MODEL_NAME', 'llama3'))

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def generate_answer(query: str, context: str = ""):
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
    else:
        prompt = query

    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
