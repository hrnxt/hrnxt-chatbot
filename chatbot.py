# chatbot.py
# FAST MODE optimized version — FIXED indexing lifecycle + SAFE EMBEDDING BATCHING

import os, io, re, hashlib, zipfile, urllib.parse, requests, logging
from typing import List, Dict, Any
import threading, traceback

# ----------------------------
# Telemetry + logging
# ----------------------------

os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ----------------------------
# Imports
# ----------------------------

import chromadb
import tiktoken
from openai import OpenAI
from pypdf import PdfReader
from chromadb.utils import embedding_functions

# ----------------------------
# Indexing state
# ----------------------------

INDEX_READY = False
INDEX_ERROR = None
_INDEX_THREAD_STARTED = False

# ----------------------------
# Config
# ----------------------------

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

DROPBOX_KB_URL = (os.environ.get("DROPBOX_KB_URL") or "").strip()
DROPBOX_EXTERNAL_URL = (os.environ.get("DROPBOX_EXTERNAL_URL") or "").strip()

# FAST MODE tuning
MAX_KB_HITS = 3
MAX_CHARS_PER_CHUNK = 800
ANSWER_CACHE_MAX = 200

# Embedding safety
UPSERT_BATCH_SIZE = 64   # critical fix

# ----------------------------
# Clients
# ----------------------------

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
)

try:
    chroma_client = chromadb.EphemeralClient()
    chroma_mode = "EphemeralClient"
except Exception:
    chroma_client = chromadb.PersistentClient(path="./chroma")
    chroma_mode = "PersistentClient"

# ----------------------------
# Collections
# ----------------------------

kb_col = chroma_client.get_or_create_collection(
    "dropbox_kb", embedding_function=openai_ef
)
external_col = chroma_client.get_or_create_collection(
    "dropbox_external", embedding_function=openai_ef
)

# ----------------------------
# Helpers
# ----------------------------

_enc = tiktoken.get_encoding("cl100k_base")

def _chunk_text(text: str, max_tokens: int = 700) -> List[str]:
    toks = _enc.encode(text or "")
    return [
        _enc.decode(toks[i:i + max_tokens])
        for i in range(0, len(toks), max_tokens)
    ]

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

# ----------------------------
# Dropbox indexing
# ----------------------------

def _download_dropbox_zip(url: str) -> bytes:
    if not url:
        return b""
    if "dl=" in url:
        url = re.sub(r"dl=\d", "dl=1", url)
    else:
        url += "&dl=1" if "?" in url else "?dl=1"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def _read_pdf(b: bytes) -> str:
    try:
        return "\n".join(
            p.extract_text() or ""
            for p in PdfReader(io.BytesIO(b)).pages
        )
    except Exception:
        return ""

def _extract_zip(zip_bytes: bytes) -> List[Dict[str, Any]]:
    docs = []
    if not zip_bytes:
        return docs
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    for info in zf.infolist():
        if info.is_dir():
            continue
        raw = zf.read(info)
        name = info.filename
        text = (
            _read_pdf(raw)
            if name.lower().endswith(".pdf")
            else raw.decode("utf-8", "ignore")
        )
        if text.strip():
            docs.append({"name": name, "text": text})
    return docs

# ----------------------------
# SAFE BATCHED UPSERT (FIX)
# ----------------------------

def _safe_upsert(col, docs, ns):
    ids, texts, metas = [], [], []

    for d in docs:
        for i, c in enumerate(_chunk_text(d["text"])):
            ids.append(_hash(f"{ns}:{d['name']}:{i}"))
            texts.append(c)
            metas.append({"source": d["name"], "chunk": i})

            # Flush batch
            if len(ids) >= UPSERT_BATCH_SIZE:
                col.upsert(ids=ids, documents=texts, metadatas=metas)
                ids, texts, metas = [], [], []

    # Final flush
    if ids:
        col.upsert(ids=ids, documents=texts, metadatas=metas)

def build_or_update_indexes():
    print(f"Chroma mode: {chroma_mode}")

    if DROPBOX_KB_URL:
        kb_docs = _extract_zip(_download_dropbox_zip(DROPBOX_KB_URL))
        print(f"KB docs parsed: {len(kb_docs)}")
        _safe_upsert(kb_col, kb_docs, "kb")

    if DROPBOX_EXTERNAL_URL:
        ex_docs = _extract_zip(_download_dropbox_zip(DROPBOX_EXTERNAL_URL))
        print(f"External docs parsed: {len(ex_docs)}")
        _safe_upsert(external_col, ex_docs, "external")

# ----------------------------
# Background indexing
# ----------------------------

def start_indexing_background():
    global INDEX_READY, INDEX_ERROR, _INDEX_THREAD_STARTED

    if _INDEX_THREAD_STARTED:
        return
    _INDEX_THREAD_STARTED = True

    def _run():
        global INDEX_READY, INDEX_ERROR
        try:
            build_or_update_indexes()
            INDEX_READY = True
            print("[INFO] Indexing complete. INDEX_READY=True")
        except Exception as e:
            INDEX_ERROR = f"{type(e).__name__}: {e}"
            print("[ERROR] Indexing failed:", INDEX_ERROR)
            print(traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()

# ----------------------------
# Retrieval
# ----------------------------

def retrieve_kb(q: str) -> List[Dict[str, Any]]:
    res = kb_col.query(query_texts=[q], n_results=MAX_KB_HITS)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [
        {"text": d[:MAX_CHARS_PER_CHUNK], "meta": m}
        for d, m in zip(docs, metas)
    ]

# ----------------------------
# Answer cache
# ----------------------------

ANSWER_CACHE: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# Main answer generation
# ----------------------------

def generate_answer(question: str) -> Dict[str, Any]:
    qkey = question.strip().lower()
    if qkey in ANSWER_CACHE:
        return ANSWER_CACHE[qkey]

    kb_hits = retrieve_kb(question)
    kb_context = "\n\n".join(h["text"] for h in kb_hits)

    system = (
        "You are Ask Mike, the HRNXT executive assistant. "
        "Be concise, practical, and grounded in HRNXT research."
    )

    user = f"""
Question:
{question}

HRNXT context:
{kb_context}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer_text = (resp.choices[0].message.content or "").strip()

    result = {
        "answer": answer_text,
        "kb_hits": [
            {
                "source": h["meta"].get("source"),
                "chunk": h["meta"].get("chunk"),
                "text": h["text"][:500],
            }
            for h in kb_hits
        ],
        "web_domain_snippets": [],
        "web_people_snippets": [],
        "g_tags": [],
    }

    if len(ANSWER_CACHE) >= ANSWER_CACHE_MAX:
        ANSWER_CACHE.pop(next(iter(ANSWER_CACHE)))
    ANSWER_CACHE[qkey] = result

    return result
