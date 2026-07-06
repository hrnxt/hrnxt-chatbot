from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chatbot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hrnxt.co",
        "https://www.hrnxt.co",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None


@app.on_event("startup")
def _startup():
    chatbot.start_indexing_background()


@app.post("/chat")
def chat(q: ChatRequest):
    if chatbot.INDEX_ERROR:
        raise HTTPException(
            status_code=503,
            detail=f"Indexing failed: {chatbot.INDEX_ERROR}"
        )

    if not chatbot.INDEX_READY:
        raise HTTPException(
            status_code=503,
            detail="Indexing in progress. Please retry in 30–90 seconds."
        )

    if q.messages:
        return chatbot.generate_answer_from_messages(
            [m.model_dump() for m in q.messages]
        )

    if q.question:
        return chatbot.generate_answer(q.question)

    raise HTTPException(
        status_code=400,
        detail="Please provide either question or messages."
    )
