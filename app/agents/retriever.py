import ollama
from pgvector.psycopg2 import register_vector
from app.config import EMBEDDING_MODEL
from app.database import get_db_connection

def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Retrieves top_k context chunks matching query vector distance."""
    # 1. Embed query
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    query_vector = response["embedding"]

    # 2. Query Postgres pgvector
    conn = get_db_connection()
    register_vector(conn)
    cursor = conn.cursor()

    # PostgreSQL <=> operator represents Cosine Distance
    cursor.execute(
        """
        SELECT content, metadata, (embedding <=> %s::vector) AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s;
        """,
        (query_vector, top_k),
    )
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return results

if __name__ == "__main__":
    # Test query against stored data
    test_query = "What is Project Alpha?"
    docs = search_documents(test_query)
    print(f"Query: {test_query}\n")
    for d in docs:
        print(f"Distance: {d['distance']:.4f}")
        print(f"Content: {d['content']}\n")
