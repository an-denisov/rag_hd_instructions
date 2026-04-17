import importlib
import sys

modules = [
    "sentence_transformers",
    "chromadb",
    "numpy",
    "fastapi",
    "uvicorn",
    "requests",
    "pydantic",
    "streamlit",
    "hf_xet",
]

print("Python:", sys.version, "\n")

for m in modules:
    try:
        importlib.import_module(m)
        print(f"[OK]   {m}")
    except ImportError:
        print(f"[MISS] {m} — НЕ установлен")
        print(f"👉 Установить:  pip install {m} --trusted-host pypi.org --trusted-host files.pythonhosted.org")
