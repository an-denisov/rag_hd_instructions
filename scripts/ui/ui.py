import streamlit as st
import requests
import json
from pathlib import Path
import uuid
import logging

logging.basicConfig(level=logging.INFO)

# ---------- Чтение конфига ----------
CONFIG_PATH = Path(__file__).parent.parent.parent / "bin" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

API_URL = f"http://{config['uvicorn']['host']}:{config['uvicorn']['port']}/ask"

# ----------- Сессия пользователя -----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id

# ---------- Настройка страницы ----------
st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Внутренняя справочная система (RAG через FastAPI)")
st.markdown("---")

# ----------- ИСТОРИЯ ----------- 
if "history" not in st.session_state:
    st.session_state.history = []

# ----------- ОТОБРАЖЕНИЕ ЧАТА ----------
for role, text in st.session_state.history:
    if role == "user":
        st.markdown(f"**Вы:** {text}")
    else:
        st.markdown(f"**ИИ:** {text}")

st.markdown("---")

# ----------- ФОРМА С ОЧИСТКОЙ ПОЛЯ ---------
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Введите вопрос:")
    submitted = st.form_submit_button("Отправить")

if submitted and question.strip():
    try:
        logging.info(f"Session ID: {SESSION_ID} | Вопрос: {question}")
        resp = requests.post(
            API_URL,
            json={"session_id": SESSION_ID, "question": question},
            timeout=300
        )
        resp.raise_for_status()
        answer = resp.json()["answer"]
        logging.info(f"Session ID: {SESSION_ID} | Ответ: {answer}")
    except Exception as e:
        answer = f"Ошибка запроса: {e}"
        logging.info(f"Session ID: {SESSION_ID} | Ответ: {answer}")

    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))
    st.rerun()
