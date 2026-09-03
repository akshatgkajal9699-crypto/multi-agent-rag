import os
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import app.config as config
from app.agents.router import run_rag_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("app/static/index.html"):
        return FileResponse("app/static/index.html")
    return {"status": "API is running. Frontend static file not found."}

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
elif os.path.exists("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.post("/query")
@app.post("/api/query")
@app.post("/chat")
@app.post("/api/chat")
async def handle_query(request: QueryRequest):
    try:
        result = run_rag_pipeline(request.query)
        return result
    except Exception as e:
        return {
            "mode": "error",
            "answer": f"Error running pipeline: {str(e)}"
        }
