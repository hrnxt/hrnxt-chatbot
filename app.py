from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import generate_answer  # this comes from your guide

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Question):
    return generate_answer(q.question)
