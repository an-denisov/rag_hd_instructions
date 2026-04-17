from fastapi import FastAPI
from pydantic import BaseModel
from .rag_core import run_rag
import threading

app = FastAPI()

# session_id → history
sessions = {}
lock = threading.Lock()   # потоки uvicorn — нужны

class AskRequest(BaseModel):
    session_id: str
    question: str

class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    session_id = req.session_id

    with lock:
        history = sessions.get(session_id, [])

    # RAG
    answer, new_history = run_rag(req.question, history)

    with lock:
        sessions[session_id] = new_history

    return AskResponse(answer=answer)
