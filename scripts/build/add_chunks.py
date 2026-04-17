import chromadb
from sentence_transformers import SentenceTransformer
import glob
import os
import json
from pathlib import Path

# ---------- Чтение конфига ----------
CONFIG_PATH = Path(__file__).parent.parent.parent / "bin" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

CHUNK_DIR = Path(__file__).parent.parent.parent / Path(config["rag"]["chunks_dir"])
CHROMA_PATH = Path(__file__).parent.parent.parent / config["chroma"]["path"] 
EMBEDDER_PATH = Path(config["models"]["embedder"])

# ---------- Создание клиента ----------
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name="chunks",
    metadata={"hnsw:space": "cosine"}
)

# ---------- Загрузка модели ----------
model = SentenceTransformer(str(EMBEDDER_PATH))

# ---------- Обработка файлов ----------
files = glob.glob(str(CHUNK_DIR / "*.md"))

print(f""" Обработка каталога {CHUNK_DIR} :""")
for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()

    filename = os.path.basename(filepath)
    doc_id = filename  # один ID = один файл

    emb = model.encode(text).tolist()

    collection.upsert(
        ids=[doc_id],
        embeddings=[emb],
        documents=[text],
        metadatas=[{"filename": filename}]
    )

    print("Добавлен:", filename)

print("Готово.")
