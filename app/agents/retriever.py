import ollama
import app.config as config

OLLAMA_HOST = getattr(config, 'OLLAMA_HOST', getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
EMBEDDING_MODEL = getattr(config, 'EMBEDDING_MODEL', getattr(config, 'EMBED_MODEL', 'nomic-embed-text'))

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def get_embedding(text: str):
    response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]
