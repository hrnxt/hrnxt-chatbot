from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chatbot  # IMPORTANT: import the module, not individual variables

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

class Question(BaseModel):
    question: str


@app.on_event("startup")
def _startup():
    # Kick off indexing as soon as the container starts
    chatbot.start_indexing_background()


@app.post("/chat")
def chat(q: Question):
    # If indexing failed, surface a readable error
    if chatbot.INDEX_ERROR:
        raise HTTPException(
            status_code=503,
            detail=f"Indexing failed: {chatbot.INDEX_ERROR}"
        )

    # If indexing still running, tell caller to retry
    if not chatbot.INDEX_READY:
        raise HTTPException(
            status_code=503,
            detail="Indexing in progress. Please retry in 30–90 seconds."
        )

    # Once ready, answer normally
    return chatbot.generate_answer(q.question)
