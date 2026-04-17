
# rag_core.py
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Tuple
from pathlib import Path
import json

# ---------- Чтение конфига ----------
CONFIG_PATH = Path(__file__).parent.parent.parent / "bin" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# ---- config ----
CHROMA_PATH = Path(__file__).parent.parent.parent / config["chroma"]["path"]           # путь к базе Chroma
COLLECTION_NAME = "chunks"
KOBOLD_URL = f"http://{config['kobold']['host']}:{config['kobold']['port']}/api/v1/generate"

# ---- load models ----
embedder_path = config["models"]["embedder"]
embedder = SentenceTransformer(embedder_path)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

History = List[Tuple[str, str]]


# =============== LOW LEVEL =================

def rewrite_question(question: str, history: History) -> str:
    conv_parts = []
    for role, text in history[-6:]:
        conv_parts.append(f"<|im_start|>{role}\n{text}\n<|im_end|>")

    conv_block = "\n".join(conv_parts)

    prompt = f"""<|im_start|>system
Перепиши последний вопрос пользователя так, чтобы он был понятен без контекста диалога.
Обязательные правила:
- Сохрани точный смысл.
- Если запрос не на русском, то переведи его на русский
- Если в последнем вопросе есть местоимения, ты ОБЯЗАН заменить их на конкретные сущности из диалога
- Если по диалогу понятно, что речь идёт о системе с конкретным названием, используй именно это название.
- Если контекст вообще не помогает (вопрос несвязан или просто мат/шутка) — верни вопрос без изменений.
- Верни только ОДНУ финальную строку-вопрос на русском, без кавычек, без комментариев и пояснений.
<|im_end|>
{conv_block}
<|im_start|>user
Переформулируй мой последний вопрос так, чтобы он был понятен без контекста диалога.

Мой последний вопрос: {question}
<|im_end|>
<|im_start|>assistant
"""

    payload = {
        "prompt": prompt,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "max_new_tokens": 64,
        "min_p": 0.02,
        "stop_sequence": [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "</s>",
        ],
        "stream": False,
    }

    resp = requests.post(KOBOLD_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    text = data["results"][0]["text"].strip()

    # Берём только первую строку
    text = text.splitlines()[0].strip()

    return clean_llm_output(text)


def search_chunks(question: str, history: History, top_k=3) -> list[str]:
    rewritten = rewrite_question(question, history)
    q_emb = embedder.encode(rewritten).tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    found_docs = res["documents"][0]
    distances  = res["distances"][0]

    # Фильтруем по distance ≤ 0.66
    filtered_docs = [
        f"distance={dist:.4f}  {doc}" 
        for doc, dist in zip(found_docs, distances) 
        if dist <= 0.66
    ]

    if not filtered_docs:
        filtered_docs = ["Не найдено в базе знаний"]

    print(f"""=== Поиск документов ===
        Входящий вопрос: 
        {question}
        Переписанный вопрос: 
        {rewritten}
        Найденные документы:""")

    for i, doc in enumerate(filtered_docs, 1):
        print(f"{i}. {doc}")

    return filtered_docs


def generate_answer(question: str, docs: list[str]) -> str:
    ctx = "\n\n".join(docs)

    prompt = f"""<|im_start|>system
Ты — внутренняя справочная система магазина. Отвечай только на основе предоставленных инструкций, кратко. 
- Если точного ответа нет — честно скажи «Не найдено в базе знаний».
- Отвечай только на русском языке, если пользователь обращается на русском. 
- Отвечай только на казахском языке, если пользователь обращается на казахском.
- Сохрани точный смысл.
- Отвечай только на основе инструкций ниже.
- Каждое утверждение в ответе должно напрямую следовать из текста инструкций.
- Не добавляй новой информации. 
- Отвечай развернуто.
<|im_end|>
<|im_start|>user
Контекст из базы знаний:
{ctx}

Вопрос сотрудника: {question}<|im_end|>
<|im_start|>assistant
"""

    payload = {
        "prompt": prompt,
        "temperature": 0.1,           # почти детерминировано
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.08,
        "max_new_tokens": 256,
        "min_p": 0.02,
        "stop_sequence": [            # KoboldCpp понимает именно stop_sequence
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<|im_start|>"
        ],
        "stream": False
    }

    response = requests.post(KOBOLD_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    answer = data["results"][0]["text"]

    return clean_llm_output(answer)


def verify_answer(question: str, docs: list[str], draft: str) -> str:
    ctx = "\n\n".join(docs)

    prompt = f"""<|im_start|>system
Ты — модуль проверки ответов в системе RAG.
Задача: проверить, насколько черновой ответ ассистента строго основан на инструкциях.
Правила:
- Используй ТОЛЬКО текст из инструкций ниже.
- НЕЛЬЗЯ добавлять новые факты или делать логические догадки.
- Если часть ответа не подтверждается инструкциями — удали её.
- Если после удаления неподтверждённых частей не остаётся полезной информации — верни ровно: Не найдено в базе знаний
- Ответ должен быть по-русски, без ссылок на эти правила, без пояснений, без форматирования и без цитирования промпта.
<|im_end|>
<|im_start|>user
Инструкции (контекст):
{ctx}

Вопрос сотрудника:
{question}

Черновой ответ ассистента:
{draft}
<|im_end|>
<|im_start|>assistant
"""

    payload = {
        "prompt": prompt,
        "temperature": 0.0,      # максимально детерминированно
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 256,
        "min_p": 0.0,
        "stop_sequence": [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "</s>",
        ],
        "stream": False,
    }
    
    resp = requests.post(KOBOLD_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    final = data["results"][0]["text"].strip()

    return clean_llm_output(final)
    
def clean_llm_output(text: str) -> str:
    """Обрезает служебные маркеры LLM и лишние строки в начале/конце."""
    if not text:
        return ""
    
    # Убираем все известные маркеры
    for marker in ["<|im_start|", "<|im_end|>", "Вопрос сотрудника:", "Контекст из базы"]:
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]

    # Обрезаем пустые строки по краям
    text = text.strip()
    
    return text    


# ================== HIGH LEVEL ===================

def run_rag(question: str, history: History | None = None) -> tuple[str, History]:
    history = history or []

    docs = search_chunks(question, history)
    draft = generate_answer(question, docs)
    final = verify_answer(question, docs, draft)

    new_hist = history + [("user", question), ("assistant", final)]
    return final, new_hist
