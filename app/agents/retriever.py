import ollama
import app.config as config

OLLAMA_HOST = getattr(config, 'OLLAMA_HOST', getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434'))
EMBEDDING_MODEL = getattr(config, 'EMBEDDING_MODEL', getattr(config, 'EMBED_MODEL', 'nomic-embed-text'))

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

def get_embedding(text: str):
    try:
        response = client.embed(model=EMBEDDING_MODEL, input=text)
        return response["embeddings"][0]
    except Exception:
        # Fallback to older SDK method format
        response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
