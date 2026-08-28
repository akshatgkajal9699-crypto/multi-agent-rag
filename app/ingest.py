import os
import glob
import ollama
from pgvector.psycopg2 import register_vector
from app.config import EMBEDDING_MODEL
from app.database import get_db_connection

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping word chunks."""
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
    """Generates a 768-dim vector embedding using Ollama."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def ingest_data():
    conn = get_db_connection()
    register_vector(conn)
    cursor = conn.cursor()

    data_files = glob.glob("data/*.txt")
    if not data_files:
        print("No text files found in data/ directory.")
        return

    for file_path in data_files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)
        print(f"Generated {len(chunks)} chunk(s).")

        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            metadata = {"source": filename, "chunk_index": idx}
            
            cursor.execute(
                """
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s);
                """,
                (chunk, psycopg2.extras.Json(metadata), embedding),
            )

    conn.commit()
    cursor.close()
    conn.close()
    print("Ingestion complete! Chunks successfully embedded and stored in Railway Postgres.")

if __name__ == "__main__":
    import psycopg2.extras
    ingest_data()
