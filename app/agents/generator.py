import ollama
from app.config import OLLAMA_HOST, LLM_MODEL

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
