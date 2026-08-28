from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anyio
from app.agents.router import route_and_execute

app = FastAPI(
    title="Multi-Agent RAG Assistant",
    description="A multi-agent RAG system powered by pgvector, Ollama, and FastAPI.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    mode: str
    answer: str
    sources: list

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        result = await anyio.to_thread.run_sync(route_and_execute, request.query)
        
        clean_sources = []
        for src in result.get("sources", []):
            if isinstance(src, dict):
                clean_sources.append({
                    "content": src.get("content", ""),
                    "metadata": dict(src.get("metadata", {})),
                    "distance": float(src.get("distance", 0.0))
                })
            else:
                # Fallback if src is returned as a tuple or row object
                clean_sources.append({
                    "content": str(src[0]) if len(src) > 0 else "",
                    "metadata": dict(src[1]) if len(src) > 1 and isinstance(src[1], dict) else {},
                    "distance": float(src[2]) if len(src) > 2 else 0.0
                })

        return QueryResponse(
            mode=result["mode"],
            answer=result["answer"],
            sources=clean_sources
        )
    except Exception as e:
        import traceback
        print("\n--- ERROR TRACEBACK ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
