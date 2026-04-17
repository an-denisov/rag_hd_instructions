import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="../chroma_db"
))

client.get_or_create_collection(
    name="chunks",
    metadata={"hnsw:space": "cosine"}
)

print("OK: коллекция создана")
