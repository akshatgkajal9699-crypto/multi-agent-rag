import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Elasticsearch Configuration
ELASTIC_URL = os.getenv("https://my-elasticsearch-project-dc1db4.es.us-central1.gcp.elastic.cloud:443", "http://localhost:9200")
ELASTIC_API_KEY = os.getenv("TnhNZVI2QUI4ZHlGYWh2eGx2QXY6cUpVaDdWQ0ZZbzFpcjVSLWZDMndqZw==", "")
