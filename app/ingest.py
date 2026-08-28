import os
import glob
import psycopg2.extras
import ollama
from pgvector.psycopg2 import register_vector
from app.config import EMBEDDING_MODEL
from app.database import get_db_connection
from app.agents.hybrid_retriever import es_client, INDEX_NAME, init_es_index

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def ingest_data():
    init_es_index()
    conn = get_db_connection()
    register_vector(conn)
    cursor = conn.cursor()

    data_files = glob.glob("data/*.txt")
    if not data_files:
        print("No text files found in data/ directory.")
        return

    for file_path in data_files:
        filename = os.path.basename(file_path)
        print(f"Processing file for Hybrid Sync: {filename}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)

        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            metadata = {"source": filename, "chunk_index": idx}
            
            # 1. PostgreSQL pgvector
            cursor.execute(
                """
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s);
                """,
                (chunk, psycopg2.extras.Json(metadata), embedding),
            )

            # 2. Elasticsearch BM25
            es_client.index(
                index=INDEX_NAME,
                body={
                    "content": chunk,
                    "source": filename,
                    "chunk_index": idx
                }
            )

    conn.commit()
    cursor.close()
    conn.close()
    print("Hybrid Ingestion Complete! Data indexed in both pgvector and Elasticsearch.")

if __name__ == "__main__":
    ingest_data()
