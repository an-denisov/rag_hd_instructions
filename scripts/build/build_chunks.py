import os
import re
import json
from pathlib import Path

# ---------- Чтение конфига ----------
CONFIG_PATH = Path(__file__).parent.parent.parent / "bin" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

CHUNK_DIR = Path(__file__).parent.parent.parent / Path(config["rag"]["chunks_dir"])
SOURCE_DIR = Path(__file__).parent.parent.parent / config["rag"]["source_dir"] 

HEADER_RE = re.compile(r"^#(.+)$", re.MULTILINE)

def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-zа-я0-9]+", "_", text)
    return text.strip("_")

def split_into_chunks(content):
    chunks = []
    current_title = None
    current_lines = []

    for line in content.splitlines():
        header_match = HEADER_RE.match(line)
        if header_match:
            if current_title and current_lines:
                chunks.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = header_match.group(1).strip()  # без # и лишних пробелов
        else:
            if current_title:
                current_lines.append(line)

    if current_title and current_lines:
        chunks.append((current_title, "\n".join(current_lines).strip()))

    return chunks

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = split_into_chunks(content)
    print(f"{os.path.basename(path)} → найдено чанков: {len(chunks)}")

    for title, text in chunks:
        slug = slugify(title)
        filename = f"{slug}.md"
        file_path = os.path.join(CHUNK_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as out:
            out.write(f"{title}\n\n{text}")

        print("  ✔ сохранён:", filename)

def main():
    if not os.path.exists(CHUNK_DIR):
        os.makedirs(CHUNK_DIR)

    for f in os.listdir(SOURCE_DIR):
        if f.endswith(".txt") or f.endswith(".md"):
            process_file(os.path.join(SOURCE_DIR, f))

if __name__ == "__main__":
    main()
