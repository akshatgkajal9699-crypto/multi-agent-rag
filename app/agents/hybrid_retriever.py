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
        print(f"Elasticsearch index '{INDEX_NAME}' created successfully.")

def bm25_search(query: str, top_k: int = 3) -> list[dict]:
    try:
        response = es_client.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": top_k
            }
        )
        results = []
        for hit in response['hits']['hits']:
            results.append({
                "content": hit['_source']['content'],
                "metadata": {
                    "source": hit['_source']['source'],
                    "chunk_index": hit['_source']['chunk_index']
                },
                "score": hit['_score']
            })
        return results
    except Exception as e:
        print(f"BM25 Search failed (falling back to pgvector): {e}")
        return []

def reciprocal_rank_fusion(vector_docs: list, bm25_docs: list, k: int = 60, top_k: int = 3) -> list:
    scores = {}

    def add_ranks(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            content = doc.get("content") if isinstance(doc, dict) else doc[0]
            if content not in scores:
                scores[content] = {"doc": doc, "rrf_score": 0.0}
            scores[content]["rrf_score"] += weight * (1.0 / (k + rank + 1))

    add_ranks(vector_docs, weight=1.0)
    add_ranks(bm25_docs, weight=1.0)

    sorted_docs = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:top_k]]

def hybrid_search(query: str, top_k: int = 3) -> list:
    vec_results = pgvector_search(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    if not bm25_results:
        return vec_results

    return reciprocal_rank_fusion(vec_results, bm25_results, top_k=top_k)

if __name__ == "__main__":
    init_es_index()
    print("Hybrid retriever initialized successfully.")
