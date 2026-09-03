import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import app.config as config
from app.agents.router import run_rag_pipeline

OLLAMA_HOST = getattr(config, 'OLLAMA_HOST', getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434'))

# Initialize Ollama client with required Ngrok bypass headers
client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
@app.post("/api/query")
@app.post("/chat")
@app.post("/api/chat")
async def handle_query(request: QueryRequest):
    try:
        result = run_rag_pipeline(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
