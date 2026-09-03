import ollama
from app.config import OLLAMA_HOST, EMBEDDING_MODEL

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def get_embedding(text: str):
    response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]
