import os
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import app.config as config
from app.agents.router import run_rag_pipeline

OLLAMA_HOST = getattr(config, 'OLLAMA_HOST', getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434'))

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Serve UI at root
@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("app/static/index.html"):
        return FileResponse("app/static/index.html")
    return {"status": "API is running. UI file not found."}

# Mount static assets if static folder exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
elif os.path.exists("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Endpoints handling chat queries
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
