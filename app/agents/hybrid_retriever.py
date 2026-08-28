import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from app.agents.retriever import search_documents as pgvector_search

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")

if ELASTIC_API_KEY:
    es_client = Elasticsearch(ELASTIC_URL, api_key=ELASTIC_API_KEY)
else:
    es_client = Elasticsearch(ELASTIC_URL)

INDEX_NAME = "rag_documents"

def init_es_index():
    if not es_client.indices.exists(index=INDEX_NAME):
        es_client.indices.create(
            index=INDEX_NAME,
            body={
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "source": {"type": "keyword"},
                        "chunk_index": {"type": "integer"}
                    }
                }
            }
        )

def bm25_search(query: str, top_k: int = 3) -> list[dict]:
    try:
        response = es_client.search(
            index=INDEX_NAME,
            body={"query": {"match": {"content": query}}, "size": top_k}
        )
        return [
            {
                "content": hit['_source']['content'],
                "metadata": {
                    "source": hit['_source']['source'],
                    "chunk_index": hit['_source']['chunk_index']
                }
            }
            for hit in response['hits']['hits']
        ]
    except Exception:
        return []

def hybrid_search(query: str, top_k: int = 3) -> list[dict]:
    raw_vec = pgvector_search(query, top_k=top_k)
    vec_results = []
    
    # Normalize pgvector outputs into structured dicts
    for item in raw_vec:
        if isinstance(item, tuple):
            vec_results.append({"content": item[0], "metadata": item[1] if len(item) > 1 else {"source": "doc", "chunk_index": 0}})
        elif isinstance(item, dict):
            vec_results.append(item)
        else:
            vec_results.append({"content": str(item), "metadata": {"source": "sample.txt", "chunk_index": 0}})

    bm25_results = bm25_search(query, top_k=top_k)
    combined = vec_results + bm25_results
    
    # Deduplicate while preserving structure
    seen = set()
    unique_docs = []
    for doc in combined:
        if doc["content"] not in seen:
            seen.add(doc["content"])
            unique_docs.append(doc)

    return unique_docs[:top_k]
