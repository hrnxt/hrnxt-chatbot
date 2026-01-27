from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import generate_answer, start_indexing_background, INDEX_READY, INDEX_ERROR

app = FastAPI()

class Question(BaseModel):
    question: str

@app.on_event("startup")
def _startup():
    # Kick off indexing as soon as the container starts
    start_indexing_background()

@app.post("/chat")
def chat(q: Question):
    # If indexing failed, surface a readable error
    if INDEX_ERROR:
        raise HTTPException(status_code=503, detail=f"Indexing failed: {INDEX_ERROR}")

    # If indexing still running, tell caller to retry
    if not INDEX_READY:
        raise HTTPException(status_code=503, detail="Indexing in progress. Please retry in 30–90 seconds.")

    # Once ready, answer normally
    return generate_answer(q.question)
