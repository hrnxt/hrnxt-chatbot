from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import generate_answer  # this comes from your guide

app = FastAPI()

class Question(BaseModel):
    question: str

from fastapi import HTTPException

@app.post("/chat")
def chat(q: Question):
    try:
        return generate_answer(q.question)
    except RuntimeError as e:
        # Friendly error for cases like “OpenAI quota exceeded” during indexing
        raise HTTPException(status_code=503, detail=str(e))
