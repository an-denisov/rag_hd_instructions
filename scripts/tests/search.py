import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

client = chromadb.Client()
collection = client.get_collection("chunks")

query = input("Введите запрос: ")

emb = model.encode(query).tolist()

res = collection.query(
    query_embeddings=[emb],
    n_results=3
)

for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    print("-----")
    print("Файл:", meta["filename"])
    print(doc[:300], "...")
