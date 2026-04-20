# 📚 RAG система (KoboldCpp + FastAPI + Streamlit)

Внутренняя справочная система с поиском по базе знаний (RAG).  
Отвечает **строго на основе загруженных документов**.

---

## ⚙️ Стек

- LLM: KoboldCpp (gguf)
- Backend: FastAPI (uvicorn)
- UI: Streamlit
- Векторная БД: ChromaDB
- Эмбеддинги: sentence-transformers (bge-m3)

---

## 📁 Структура проекта

bin/
  start.ps1          # запуск LLM + API + UI
  config.json        # конфигурация

scripts/
  build/             # подготовка данных (НЕ сервис)
    build_chunks.py  # разбивка исходников на чанки
    add_chunks.py    # загрузка чанков в Chroma
    init_chroma.py   # (опционально) создание БД

  server/
    server.py        # FastAPI API (/ask)
    rag_core.py      # вся логика RAG

  ui/
    ui.py            # Streamlit интерфейс

  tests/             # тесты и утилиты

---

## ⚙️ Конфигурация (bin/config.json)

Пример:

{
  "models": {
    "embedder": "C:/AI/models/bge-m3",
    "llm": "C:/AI/models/qwen2.5-7b-instruct-q4_k_m/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
  },
  "kobold": {
    "exe": "C:/AI/kobold/koboldcpp.exe",
    "host": "127.0.0.1",
    "port": 5001
  },
  "uvicorn": {
    "host": "127.0.0.1",
    "port": 8000
  },
  "streamlit": {
    "host": "0.0.0.0",
    "port": 8501
  },
  "chroma": {
    "path": "chroma_db"
  },
  "rag": {
    "chunks_dir": "chunks",
    "source_dir": "sources"
  }
}

---

## 🌐 Порты и доступ

| Сервис     | URL                         | Описание              |
|-----------|-----------------------------|----------------------|
| KoboldCpp | http://127.0.0.1:5001       | LLM API              |
| FastAPI   | http://127.0.0.1:8000/ask   | Backend (RAG)        |
| UI        | http://localhost:8501       | Веб-интерфейс        |

---

## 🚀 Полный запуск (по шагам)

### 1. Установка зависимостей

pip install sentence-transformers chromadb fastapi uvicorn requests pydantic streamlit

---

## 🧪 Тесты

Проверка окружения:

python scripts/tests/check_env.py

---

## 📚 Подготовка базы знаний (ОБЯЗАТЕЛЬНО)

### 2. Положить исходные файлы

В папку:

rag.source_dir

Пример:
sources/
  instructions.txt
  faq.md

---

### 3. Разбить на чанки

python scripts/build/build_chunks.py

👉 создаст:
chunks/
  *.md

---

### 4. Загрузить чанки в базу

python scripts/build/add_chunks.py

👉 результат:
- создаётся папка `chroma_db`
- создаётся коллекция `chunks`
- все документы индексируются

---

## ▶️ Запуск системы

bin/start.ps1

⚠️ Важно:
этот скрипт запускает ТОЛЬКО:
- KoboldCpp
- FastAPI
- Streamlit

Он НЕ:
- создаёт чанки
- НЕ загружает данные

---

## 📡 API

POST /ask

Запрос:

{
  "session_id": "user1",
  "question": "Как сделать возврат?"
}

Ответ:

{
  "answer": "..."
}

---

## 🧠 Как работает RAG

1. Вопрос переписывается с учётом истории
2. Строится embedding
3. Поиск в ChromaDB
4. Генерация ответа (LLM)
5. Проверка ответа (anti-hallucination)

---

## 🔒 Ограничения

- Ответы только из базы знаний
- Если нет данных → "Не найдено в базе знаний"
- Без внешнего интернета

---

## 🛠 Особенности

- учёт истории диалога
- фильтр по distance
- двойная генерация:
  - answer
  - verify
- поддержка RU / KZ

