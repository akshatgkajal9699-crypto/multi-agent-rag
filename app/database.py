import psycopg2
from psycopg2.extras import RealDictCursor
from app.config import DATABASE_URL

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create documents table with 768 dimension vector support
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            embedding vector(768)
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully with pgvector support.")

if __name__ == "__main__":
    init_db()
