from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(path="C:/AI/rag/chroma_db")

collection = client.get_collection("chunks")

print("Документов:", collection.count())

model = SentenceTransformer("BAAI/bge-m3")

query = "Как напечатать ценники?"
q_emb = model.encode(query).tolist()

res = collection.query(
    query_embeddings=[q_emb],
    n_results=3
)

print(res)
