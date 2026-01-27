# chatbot.py
# Extracted from final_Ask_Mike_chatbot.ipynb and made production-safe (no Colab dependencies, no interactive prompts)

import os, io, re, hashlib, zipfile, tempfile, urllib.parse, requests, logging, json, time, textwrap, sys
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache

import pandas as pd

# Chroma telemetry off before import
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

import chromadb
import tiktoken
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
import html2text


# ----------------------------
# Config (from notebook)
# ----------------------------

# Secrets from environment (NO prompting in production)
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
SERPER_API_KEY = (os.environ.get("SERPER_API_KEY") or "").strip()

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL
)


# Collections
KB_COLLECTION_NAME       = "dropbox_kb"
STYLE_COLLECTION_NAME    = "dropbox_style"
EXTERNAL_COLLECTION_NAME = "dropbox_external"

TRUSTED_SOURCES = {"names": [], "domains": []}

# Defaults that were Colab paths in the notebook; made configurable for servers
CHROMA_PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma")
EXTERNAL_EXCEL_PATH = os.environ.get("EXTERNAL_EXCEL_PATH", "") or "./Chatbot Trusted Sources.xlsx"
DBX_CACHE_DIR       = os.environ.get("DBX_CACHE_DIR", "./.dbx_cache")
MANIFEST_PATH       = os.environ.get("DBX_MANIFEST_PATH", "./dbx_manifest.json")

# Retrieval and web limits (notebook rules)
KB_TOP_K        = 6
STYLE_TOP_K     = 4
WEB_TOTAL_MAX   = 5
WEB_PER_DOMAIN  = 2
WEB_MAX_PEOPLE  = 2

# Dropbox URLs (set these as environment variables in Render later)
DROPBOX_KB_URL       = (os.environ.get("DROPBOX_KB_URL") or "").strip()
DROPBOX_STYLE_URL    = (os.environ.get("DROPBOX_STYLE_URL") or "").strip()
DROPBOX_EXTERNAL_URL = (os.environ.get("DROPBOX_EXTERNAL_URL") or "").strip()


# ----------------------------
# Clients (OpenAI + Chroma)
# ----------------------------

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

try:
    chroma_client = chromadb.EphemeralClient()
    chroma_mode = "EphemeralClient (in-memory)"
except Exception:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    chroma_mode = "PersistentClient (CHROMA_PERSIST_PATH)"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------
# Helpers: file reading/parsing
# ----------------------------

def _read_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception:
        return ""

def _read_docx_bytes(b: bytes) -> str:
    try:
        doc = Document(io.BytesIO(b))
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception:
        return ""

def _read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _read_html_bytes(b: bytes) -> str:
    try:
        soup = BeautifulSoup(b, "html.parser")
        # remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return ""

def _guess_ext(name: str) -> str:
    name = (name or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(".html") or name.endswith(".htm"):
        return "html"
    if name.endswith(".txt") or name.endswith(".md"):
        return "txt"
    return ""


# ----------------------------
# Helpers: basic token chunking
# ----------------------------

_enc = tiktoken.get_encoding("cl100k_base")

def _chunk_text(text: str, max_tokens: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    toks = _enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        j = min(i + max_tokens, len(toks))
        chunk = _enc.decode(toks[i:j])
        chunks.append(chunk)
        if j == len(toks):
            break
        i = max(0, j - overlap)
    return chunks


# ----------------------------
# Dropbox + indexing (from notebook)
# ----------------------------

def _download_dropbox_zip(shared_url: str) -> bytes:
    """
    Expects a Dropbox shared link. Converts it to a direct download.
    """
    if not shared_url:
        return b""
    # Convert shared URL to direct download
    # e.g. https://www.dropbox.com/s/<id>/file.zip?dl=0 -> dl=1
    url = shared_url
    if "dropbox.com" in url and "dl=" in url:
        url = re.sub(r"dl=\d", "dl=1", url)
    elif "dropbox.com" in url and "dl=" not in url:
        sep = "&" if "?" in url else "?"
        url = url + f"{sep}dl=1"

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _extract_zip_to_docs(zip_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns list of docs: {name, text}
    """
    docs: List[Dict[str, Any]] = []
    if not zip_bytes:
        return docs
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        ext = _guess_ext(name)
        raw = zf.read(info)
        text = ""
        if ext == "pdf":
            text = _read_pdf_bytes(raw)
        elif ext == "docx":
            text = _read_docx_bytes(raw)
        elif ext == "html":
            text = _read_html_bytes(raw)
        else:
            text = _read_txt_bytes(raw)
        text = (text or "").strip()
        if text:
            docs.append({"name": name, "text": text})
    return docs


def _get_or_create_collection(name: str):
    try:
        return chroma_client.get_collection(name, embedding_function=openai_ef)
    except Exception:
        return chroma_client.create_collection(name, embedding_function=openai_ef)


kb_col       = _get_or_create_collection(KB_COLLECTION_NAME)
style_col    = _get_or_create_collection(STYLE_COLLECTION_NAME)
external_col = _get_or_create_collection(EXTERNAL_COLLECTION_NAME)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _upsert_docs(col, docs: List[Dict[str, Any]], namespace: str):
    ids = []
    texts = []
    metas = []
    for d in docs:
        name = d["name"]
        for idx, chunk in enumerate(_chunk_text(d["text"])):
            cid = _hash(f"{namespace}:{name}:{idx}:{chunk[:50]}")
            ids.append(cid)
            texts.append(chunk)
            metas.append({"source": name, "chunk": idx, "ns": namespace})

    if not ids:
        return

    try:
        col.upsert(ids=ids, documents=texts, metadatas=metas)
    except Exception as e:
        # Don’t crash the whole app if embedding/upsert fails (e.g., OpenAI 429 quota)
        # This will show up in Render -> Logs -> Application logs
        print(f"[ERROR] Chroma upsert failed for namespace='{namespace}' with {len(ids)} chunks: {repr(e)}")

        # Raise a simple RuntimeError so upstream code can handle it cleanly
        # (avoids Chroma/OpenAI wrapper TypeErrors)
        raise RuntimeError(
            "Indexing failed while embedding documents. "
            "Most commonly this is due to OpenAI API quota/billing (HTTP 429). "
            "Check OpenAI Billing / project limits, then redeploy and retry."
        ) from e


_INDEXED_ONCE = False

def build_or_update_indexes() -> None:
    """
    Pulls KB/style/external from Dropbox ZIPs (if provided) and builds Chroma collections.
    """
    global _INDEXED_ONCE
    _log(f"Chroma mode: {chroma_mode}")

    if DROPBOX_KB_URL:
        _log("Downloading KB zip…")
        kb_zip = _download_dropbox_zip(DROPBOX_KB_URL)
        kb_docs = _extract_zip_to_docs(kb_zip)
        _log(f"KB docs parsed: {len(kb_docs)}")
        _upsert_docs(kb_col, kb_docs, "kb")

    if DROPBOX_STYLE_URL:
        _log("Downloading Style zip…")
        style_zip = _download_dropbox_zip(DROPBOX_STYLE_URL)
        style_docs = _extract_zip_to_docs(style_zip)
        _log(f"Style docs parsed: {len(style_docs)}")
        _upsert_docs(style_col, style_docs, "style")

    if DROPBOX_EXTERNAL_URL:
        _log("Downloading External zip…")
        ext_zip = _download_dropbox_zip(DROPBOX_EXTERNAL_URL)
        ext_docs = _extract_zip_to_docs(ext_zip)
        _log(f"External docs parsed: {len(ext_docs)}")
        _upsert_docs(external_col, ext_docs, "external")

    _INDEXED_ONCE = True


def _ensure_indexed_once():
    global _INDEXED_ONCE
    if not _INDEXED_ONCE:
        build_or_update_indexes()


# ----------------------------
# Retrieval
# ----------------------------

def _query_collection(col, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query:
        return []
    res = col.query(query_texts=[query], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "meta": meta, "distance": dist})
    return out

def retrieve_kb(query: str, top_k: int = KB_TOP_K):
    return _query_collection(kb_col, query, top_k=top_k)

def retrieve_style(query: str, top_k: int = STYLE_TOP_K):
    return _query_collection(style_col, query, top_k=top_k)


# ----------------------------
# Web snippets (Serper)
# ----------------------------

_WEB_SNIP_CACHE: dict = {}

def _serper_search(query: str, num: int = 5) -> List[Dict[str, Any]]:
    if not SERPER_API_KEY:
        return []
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        out = []
        for item in (data.get("organic") or []):
            out.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })
        return out
    except Exception:
        return []

def _fetch_page_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        # Skip PDFs/XML/RSS like the guide indicates
        if "pdf" in ctype or "xml" in ctype or "rss" in ctype:
            return ""
        return _read_html_bytes(r.content)
    except Exception:
        return ""

def _get_web_snippets(question: str) -> Dict[str, List[Dict[str, Any]]]:
    cache_key = ("web", question.strip().lower())
    if cache_key in _WEB_SNIP_CACHE:
        return _WEB_SNIP_CACHE[cache_key]

    results = _serper_search(question, num=10)
    domain_snips = []
    people_snips = []

    # Simple heuristic split: "people" are pages where snippet includes an author-like byline
    for item in results:
        if len(domain_snips) + len(people_snips) >= WEB_TOTAL_MAX:
            break
        link = item.get("link") or ""
        snip = (item.get("snippet") or "").strip()
        if not link or not snip:
            continue

        domain = urllib.parse.urlparse(link).netloc.lower()
        text = _fetch_page_text(link)
        if not text:
            continue

        record = {"url": link, "domain": domain, "snippet": snip, "text": text[:2000]}

        # crude author/byline heuristic
        if re.search(r"\bby\s+[A-Z][a-z]+", snip) and len(people_snips) < WEB_MAX_PEOPLE:
            people_snips.append(record)
        else:
            # per-domain cap
            if sum(1 for d in domain_snips if d["domain"] == domain) >= WEB_PER_DOMAIN:
                continue
            domain_snips.append(record)

    out = {"domain": domain_snips, "people": people_snips}
    _WEB_SNIP_CACHE[cache_key] = out
    return out


# ----------------------------
# Main answer generation
# ----------------------------

def generate_answer(question: str) -> Dict[str, Any]:
    """
    Returns dict in the guide’s shape:
    - answer
    - kb_hits
    - web_domain_snippets
    - web_people_snippets
    - g_tags
    """
    if client is None:
        return {
            "answer": "OPENAI_API_KEY is not set on the server.",
            "kb_hits": [],
            "web_domain_snippets": [],
            "web_people_snippets": [],
            "g_tags": [],
        }

    _ensure_indexed_once()

    kb_hits    = retrieve_kb(question, top_k=KB_TOP_K)
    style_hits = retrieve_style(question, top_k=STYLE_TOP_K)

    web = _get_web_snippets(question)
    domain_snips = web.get("domain", [])
    people_snips = web.get("people", [])

    # Build context
    kb_context = "\n\n".join([h["text"] for h in kb_hits[:KB_TOP_K]])
    style_context = "\n\n".join([h["text"] for h in style_hits[:STYLE_TOP_K]])

    web_context = ""
    for w in domain_snips:
        web_context += f"\n\n[WEB] {w['url']}\n{w['snippet']}\n{w['text'][:1200]}"
    for w in people_snips:
        web_context += f"\n\n[PERSON] {w['url']}\n{w['snippet']}\n{w['text'][:1200]}"

    system = (
        "You are Ask Mike, the HRNXT executive assistant. "
        "Be clear, practical, and grounded. "
        "If you use web content, summarize it briefly and avoid over-quoting."
    )

    user = f"""
Question:
{question}

KB context:
{kb_context}

Style context:
{style_context}

Web context:
{web_context}
"""

    resp = client.chat.completions.create(
        model=os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer_text = (resp.choices[0].message.content or "").strip()

    # Format snippet outputs similar to guide expectations
    kb_out = []
    for h in kb_hits[:KB_TOP_K]:
        meta = h.get("meta") or {}
        kb_out.append({
            "source": meta.get("source", ""),
            "chunk": meta.get("chunk", ""),
            "text": (h.get("text") or "")[:500],
        })

    web_domain_out = [{"url": w["url"], "domain": w["domain"], "snippet": w["snippet"]} for w in domain_snips]
    web_people_out = [{"url": w["url"], "domain": w["domain"], "snippet": w["snippet"]} for w in people_snips]

    return {
        "answer": answer_text,
        "kb_hits": kb_out,
        "web_domain_snippets": web_domain_out,
        "web_people_snippets": web_people_out,
        "g_tags": [],
    }
