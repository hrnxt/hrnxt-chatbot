# chatbot.py
# HRNXT Ask Mike
# Knowledge-base upgrade:
# - PDF + DOCX + XLSX + text-file ingestion
# - source-type metadata
# - more useful/diversified retrieval context
# - executive-level response style
# - faster first-turn handling + selective follow-up rewriting
# - timing/retrieval logs

import os
import io
import re
import time
import hashlib
import zipfile
import requests
import logging
import threading
import traceback
import xml.etree.ElementTree as ET

from typing import List, Dict, Any


# ============================================================
# TELEMETRY + LOGGING
# ============================================================

os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)


# ============================================================
# IMPORTS
# ============================================================

import chromadb
import tiktoken

from openai import OpenAI
from pypdf import PdfReader
from chromadb.utils import embedding_functions


# ============================================================
# INDEXING STATE
# ============================================================

INDEX_READY = False
INDEX_ERROR = None
_INDEX_THREAD_STARTED = False


# ============================================================
# CONFIG
# ============================================================

OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or ""
).strip()

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gpt-4o-mini"
)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

DROPBOX_KB_URL = (
    os.environ.get("DROPBOX_KB_URL")
    or ""
).strip()

DROPBOX_EXTERNAL_URL = (
    os.environ.get("DROPBOX_EXTERNAL_URL")
    or ""
).strip()


# Retrieval tuning
#
# We retrieve more than the old 3 x 800-character snippets, but keep
# a firm cap so Ask Mike remains fast and focused.

MAX_KB_HITS = int(
    os.environ.get(
        "MAX_KB_HITS",
        "5"
    )
)

MAX_HITS_PER_SOURCE = int(
    os.environ.get(
        "MAX_HITS_PER_SOURCE",
        "2"
    )
)

MAX_CHARS_PER_CHUNK = int(
    os.environ.get(
        "MAX_CHARS_PER_CHUNK",
        "1600"
    )
)

MAX_CONTEXT_CHARS = int(
    os.environ.get(
        "MAX_CONTEXT_CHARS",
        "8000"
    )
)


# Keep Ask Mike from drifting into essays.

MAX_ANSWER_TOKENS = int(
    os.environ.get(
        "MAX_ANSWER_TOKENS",
        "650"
    )
)


# Answer cache

ANSWER_CACHE_MAX = 200


# Safe embedding batch size

UPSERT_BATCH_SIZE = 64


# ============================================================
# ASK MIKE SYSTEM PROMPT
# ============================================================

ASK_MIKE_SYSTEM_PROMPT = """
You are Ask Mike, an executive adviser to senior HR leaders.

Your audience is primarily CHROs, Chief People Officers, senior HR executives,
and experienced functional leaders. Assume they already understand foundational
HR concepts and do not need introductory explanations.

Your job is not to summarize a topic. Your job is to help a senior HR leader
make a better decision.

Before writing, identify the single most important judgment you would give this
executive. Lead with that judgment. Build the answer around it.

Do not begin by categorizing the topic into "key areas," "key considerations,"
"critical elements," or a generic framework unless the user explicitly asks for
a framework or structured analysis.

Respond like a thoughtful, experienced adviser or peer to a CHRO:
clear, commercially aware, pragmatic, nuanced, and willing to take a point of view.

Core response principles:

- Lead with a sharp thesis. The first one or two sentences should contain
  the most useful judgment, implication, or recommendation.
- Prefer a clear point of view over balanced-but-generic commentary.
- Be selective. Two or three substantive ideas are better than six broadly sensible ones.
- Explain the consequence behind the recommendation.
- Surface the tension: what happens if the organization goes too far in one
  direction versus the other.
- Focus on executive decisions: decision rights, accountability, sequencing,
  governance, operating model, workforce implications, execution, and tradeoffs.
- Distinguish between what should be centralized versus decentralized,
  standardized versus adapted, and decided now versus learned through iteration
  when those distinctions matter.
- When context matters, say what would change the recommendation.
- Avoid false certainty. Be decisive without pretending there is one universal answer.
- When useful, identify the one or two questions the executive team should be debating.
- End with a practical next move when there is a meaningful one.

Executive voice:

- Write for executives, not beginners.
- Sound like a senior peer, not a textbook, consultant deck, marketer, or generic AI assistant.
- Use plain English and avoid jargon unless it adds precision.
- Favor concise, memorable framing when it clarifies a tradeoff.
- Avoid motivational language and filler.
- Avoid generic phrases such as:
  "Here are some key strategies,"
  "There are several factors to consider,"
  "It's important to,"
  "Organizations should focus on,"
  "A balanced approach is needed,"
  or "In today's rapidly changing environment."
- Do not use headings such as:
  "Key Considerations,"
  "Key Elements,"
  "Critical Areas,"
  "Next Steps,"
  "Next Move,"
  "In Summary,"
  or "In Conclusion"
  unless the user explicitly asks for a framework, checklist, or structured breakdown.
- Avoid textbook definitions unless the user explicitly asks for one.
- Do not restate the user's question.
- Do not automatically produce a numbered or bulleted list.
- Use bullets only when they materially improve executive scanability.
- Do not create long frameworks when two or three substantive points are enough.
- Prefer short paragraphs and clear executive-level language.
- Keep most answers to roughly 3 to 5 short paragraphs.
- Expand only when the question genuinely requires depth.

Depth and judgment:

- Move beyond obvious advice.
- Where possible, identify the non-obvious risk, tradeoff, or second-order effect.
- If a recommendation sounds generic, make it more specific by explaining:
  who should own it, what should change, what should not change, or what could go wrong.
- For governance questions, clarify decision rights and boundaries rather than
  simply recommending "more governance."
- For operating-model questions, distinguish enterprise standards from local execution.
- For transformation questions, separate technology adoption from the organizational
  changes required to make it work.
- For workforce questions, connect the issue to capability, incentives, roles,
  trust, and management systems where relevant.
- For AI questions, distinguish experimentation, automation, decision authority,
  accountability, and human oversight rather than treating "AI strategy" as one thing.

Use of evidence and HRNXT context:

- Ground the answer in supplied HRNXT/Executive Networks research when it is relevant.
- Do not force retrieved material into the answer if it is weakly related.
- Never claim the supplied context says something it does not.
- If the evidence is limited, be appropriately cautious rather than inventing support.
- Synthesize research into judgment rather than merely repeating it.
- Material identified as "thought_leadership" may contain curated names,
  books, articles, and citations rather than the full underlying works.
  Treat that material as a map to relevant thinkers and ideas, not as proof
  that you have retrieved or read the underlying book or article.
- Do not attribute a specific finding, statistic, conclusion, or quotation
  to an underlying work unless that finding is actually present in the
  supplied retrieved context.
- When retrieved research contains a useful source name or study, you may
  identify it briefly, but the answer should remain advisory rather than
  becoming a literature review.

For strategic questions, favor this general pattern when appropriate:
1. State the central judgment.
2. Explain the key tradeoff, consequence, or organizational implication.
3. Recommend the next decision or action.

For follow-up questions:
- Build on the prior conversation rather than repeating the earlier answer.
- Move the discussion forward.
- If the user asks "which one," "what first," or "what would you do,"
  make a choice unless the available context genuinely prevents one.
"""


# ============================================================
# CLIENTS
# ============================================================

client = (
    OpenAI(
        api_key=OPENAI_API_KEY
    )
    if OPENAI_API_KEY
    else None
)

openai_ef = (
    embedding_functions
    .OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )
)


# ============================================================
# CHROMA
# ============================================================

try:

    chroma_client = (
        chromadb.EphemeralClient()
    )

    chroma_mode = "EphemeralClient"

except Exception:

    chroma_client = (
        chromadb.PersistentClient(
            path="./chroma"
        )
    )

    chroma_mode = "PersistentClient"


# ============================================================
# COLLECTIONS
# ============================================================

kb_col = (
    chroma_client
    .get_or_create_collection(
        "dropbox_kb",
        embedding_function=openai_ef
    )
)

external_col = (
    chroma_client
    .get_or_create_collection(
        "dropbox_external",
        embedding_function=openai_ef
    )
)


# ============================================================
# HELPERS
# ============================================================

_enc = tiktoken.get_encoding(
    "cl100k_base"
)


def _chunk_text(
    text: str,
    max_tokens: int = 700
) -> List[str]:

    toks = _enc.encode(
        text or ""
    )

    return [
        _enc.decode(
            toks[i:i + max_tokens]
        )
        for i
        in range(
            0,
            len(toks),
            max_tokens
        )
    ]


def _hash(
    s: str
) -> str:

    return hashlib.sha256(
        s.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


def _ensure_client():
    """
    Fail clearly if the OpenAI key is missing.
    """

    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )


def _log_timing(
    label: str,
    started_at: float
):
    """
    Simple Render-friendly timing log.
    """

    elapsed = (
        time.perf_counter()
        - started_at
    )

    print(
        f"[TIMING] {label}: "
        f"{elapsed:.2f}s"
    )


def _source_type(
    filename: str
) -> str:
    """
    Tag knowledge-base files so the answer model can distinguish
    substantive research from thought-leadership reference material.
    """

    base = os.path.basename(
        filename or ""
    ).lower()

    if base.startswith(
        "ask_mike_research_layer"
    ):
        return "research_layer"

    if base.startswith(
        "thinkers50"
    ):
        return "thought_leadership"

    if (
        "executive networks" in base
        or base.startswith("en_")
    ):
        return "en_research"

    return "knowledge_base"


# ============================================================
# DROPBOX DOWNLOAD
# ============================================================

def _download_dropbox_zip(
    url: str
) -> bytes:

    if not url:
        return b""

    if "dl=" in url:

        url = re.sub(
            r"dl=\d",
            "dl=1",
            url
        )

    else:

        url += (
            "&dl=1"
            if "?" in url
            else "?dl=1"
        )

    r = requests.get(
        url,
        timeout=60
    )

    r.raise_for_status()

    return r.content


# ============================================================
# FILE READERS
# ============================================================

def _read_pdf(
    b: bytes
) -> str:

    try:

        return "\n".join(
            page.extract_text()
            or ""
            for page
            in PdfReader(
                io.BytesIO(b)
            ).pages
        )

    except Exception as exc:

        print(
            "[WARN] PDF read failed:",
            exc
        )

        return ""


def _read_docx(
    b: bytes
) -> str:
    """
    Read ordinary .docx text using only the Python standard library.

    This intentionally avoids requiring python-docx on Render.
    It extracts paragraphs from the main document plus common
    note/header/footer XML parts when present.
    """

    try:

        zf = zipfile.ZipFile(
            io.BytesIO(b)
        )

        candidate_parts = [
            name
            for name in zf.namelist()
            if (
                name == "word/document.xml"
                or name.startswith("word/header")
                or name.startswith("word/footer")
                or name in (
                    "word/footnotes.xml",
                    "word/endnotes.xml",
                )
            )
            and name.endswith(".xml")
        ]

        paragraphs = []

        ns = {
            "w":
                "http://schemas.openxmlformats.org/"
                "wordprocessingml/2006/main"
        }

        for part_name in candidate_parts:

            root = ET.fromstring(
                zf.read(part_name)
            )

            for paragraph in root.findall(
                ".//w:p",
                ns
            ):

                pieces = []

                for node in paragraph.iter():

                    tag = (
                        node.tag.split("}")[-1]
                        if "}" in node.tag
                        else node.tag
                    )

                    if (
                        tag == "t"
                        and node.text
                    ):
                        pieces.append(
                            node.text
                        )

                    elif tag == "tab":
                        pieces.append("\t")

                    elif tag in (
                        "br",
                        "cr",
                    ):
                        pieces.append("\n")

                text = "".join(
                    pieces
                ).strip()

                if text:
                    paragraphs.append(
                        text
                    )

        return "\n".join(
            paragraphs
        )

    except Exception as exc:

        print(
            "[WARN] DOCX read failed:",
            exc
        )

        return ""


def _read_xlsx(
    b: bytes
) -> str:
    """
    Convert ordinary XLSX cell values to tab-separated text using
    only the standard library.

    This makes the existing Trusted Sources workbook indexable.
    The current answer path still does NOT treat this workbook as
    substantive evidence; it remains in external_col for future use.
    """

    try:

        zf = zipfile.ZipFile(
            io.BytesIO(b)
        )

        shared_strings = []

        if "xl/sharedStrings.xml" in zf.namelist():

            root = ET.fromstring(
                zf.read(
                    "xl/sharedStrings.xml"
                )
            )

            for si in root:

                text_bits = [
                    node.text or ""
                    for node in si.iter()
                    if (
                        node.tag.split("}")[-1]
                        == "t"
                    )
                ]

                shared_strings.append(
                    "".join(text_bits)
                )

        sheet_names = sorted(
            name
            for name in zf.namelist()
            if (
                name.startswith(
                    "xl/worksheets/sheet"
                )
                and name.endswith(".xml")
            )
        )

        output_lines = []

        for sheet_name in sheet_names:

            root = ET.fromstring(
                zf.read(sheet_name)
            )

            output_lines.append(
                f"[Worksheet: {os.path.basename(sheet_name)}]"
            )

            for row in root.iter():

                if (
                    row.tag.split("}")[-1]
                    != "row"
                ):
                    continue

                values = []

                for cell in row:

                    if (
                        cell.tag.split("}")[-1]
                        != "c"
                    ):
                        continue

                    cell_type = cell.attrib.get(
                        "t"
                    )

                    value = ""

                    if cell_type == "inlineStr":

                        text_bits = [
                            node.text or ""
                            for node in cell.iter()
                            if (
                                node.tag.split("}")[-1]
                                == "t"
                            )
                        ]

                        value = "".join(
                            text_bits
                        )

                    else:

                        v_node = None

                        for node in cell:

                            if (
                                node.tag.split("}")[-1]
                                == "v"
                            ):
                                v_node = node
                                break

                        raw_value = (
                            v_node.text
                            if (
                                v_node is not None
                                and v_node.text
                            )
                            else ""
                        )

                        if (
                            cell_type == "s"
                            and raw_value
                        ):

                            try:

                                value = (
                                    shared_strings[
                                        int(raw_value)
                                    ]
                                )

                            except Exception:

                                value = raw_value

                        else:

                            value = raw_value

                    values.append(
                        value.strip()
                    )

                if any(values):

                    output_lines.append(
                        "\t".join(values)
                    )

        return "\n".join(
            output_lines
        )

    except Exception as exc:

        print(
            "[WARN] XLSX read failed:",
            exc
        )

        return ""


def _read_text_file(
    raw: bytes
) -> str:

    return raw.decode(
        "utf-8",
        "ignore"
    )


# ============================================================
# DROPBOX ZIP EXTRACTION
# ============================================================

def _extract_zip(
    zip_bytes: bytes
) -> List[Dict[str, Any]]:

    docs = []

    if not zip_bytes:
        return docs

    zf = zipfile.ZipFile(
        io.BytesIO(
            zip_bytes
        )
    )

    for info in zf.infolist():

        if info.is_dir():
            continue

        name = info.filename

        base = os.path.basename(
            name
        )

        # Skip common hidden/temp files.
        if (
            not base
            or base.startswith("~$")
            or base.startswith(".")
            or "__MACOSX" in name
        ):
            continue

        raw = zf.read(
            info
        )

        lower_name = (
            name.lower()
        )

        if lower_name.endswith(
            ".pdf"
        ):

            text = _read_pdf(
                raw
            )

        elif lower_name.endswith(
            ".docx"
        ):

            text = _read_docx(
                raw
            )

        elif lower_name.endswith(
            ".xlsx"
        ):

            text = _read_xlsx(
                raw
            )

        elif lower_name.endswith(
            (
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".html",
                ".htm",
                ".xml",
            )
        ):

            text = _read_text_file(
                raw
            )

        else:

            # Preserve the old fallback behavior for any simple
            # text-like file types we have not explicitly listed.
            text = _read_text_file(
                raw
            )

        if text.strip():

            docs.append(
                {
                    "name":
                        name,

                    "text":
                        text,

                    "source_type":
                        _source_type(
                            name
                        ),
                }
            )

            print(
                f"[INFO] Parsed: {name} "
                f"({len(text):,} chars)"
            )

        else:

            print(
                f"[WARN] No readable text: "
                f"{name}"
            )

    return docs


# ============================================================
# SAFE BATCHED UPSERT
# ============================================================

def _safe_upsert(
    col,
    docs,
    namespace
):

    ids = []
    texts = []
    metas = []

    total_chunks = 0

    for doc in docs:

        chunks = _chunk_text(
            doc["text"]
        )

        for i, chunk in enumerate(
            chunks
        ):

            ids.append(
                _hash(
                    f"{namespace}:"
                    f"{doc['name']}:"
                    f"{i}"
                )
            )

            texts.append(
                chunk
            )

            metas.append(
                {
                    "source":
                        doc["name"],

                    "source_type":
                        doc.get(
                            "source_type",
                            "knowledge_base"
                        ),

                    "chunk":
                        i,
                }
            )

            total_chunks += 1

            if (
                len(ids)
                >= UPSERT_BATCH_SIZE
            ):

                col.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metas
                )

                ids = []
                texts = []
                metas = []

    if ids:

        col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas
        )

    return total_chunks


# ============================================================
# BUILD / UPDATE INDEXES
# ============================================================

def build_or_update_indexes():

    started = (
        time.perf_counter()
    )

    print(
        f"[INFO] Chroma mode: "
        f"{chroma_mode}"
    )


    if DROPBOX_KB_URL:

        kb_started = (
            time.perf_counter()
        )

        kb_zip = (
            _download_dropbox_zip(
                DROPBOX_KB_URL
            )
        )

        kb_docs = (
            _extract_zip(
                kb_zip
            )
        )

        print(
            f"[INFO] KB docs parsed: "
            f"{len(kb_docs)}"
        )

        kb_chunks = _safe_upsert(
            kb_col,
            kb_docs,
            "kb"
        )

        print(
            f"[INFO] KB chunks indexed: "
            f"{kb_chunks}"
        )

        _log_timing(
            "KB indexing",
            kb_started
        )


    if DROPBOX_EXTERNAL_URL:

        external_started = (
            time.perf_counter()
        )

        external_zip = (
            _download_dropbox_zip(
                DROPBOX_EXTERNAL_URL
            )
        )

        external_docs = (
            _extract_zip(
                external_zip
            )
        )

        print(
            f"[INFO] External docs parsed: "
            f"{len(external_docs)}"
        )

        external_chunks = _safe_upsert(
            external_col,
            external_docs,
            "external"
        )

        print(
            f"[INFO] External chunks indexed: "
            f"{external_chunks}"
        )

        _log_timing(
            "External indexing",
            external_started
        )


    _log_timing(
        "Total indexing",
        started
    )


# ============================================================
# BACKGROUND INDEXING
# ============================================================

def start_indexing_background():

    global INDEX_READY
    global INDEX_ERROR
    global _INDEX_THREAD_STARTED


    if _INDEX_THREAD_STARTED:
        return


    _INDEX_THREAD_STARTED = True


    def _run():

        global INDEX_READY
        global INDEX_ERROR

        try:

            build_or_update_indexes()

            INDEX_READY = True

            print(
                "[INFO] Indexing complete. "
                "INDEX_READY=True"
            )

        except Exception as exc:

            INDEX_ERROR = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "[ERROR] Indexing failed:",
                INDEX_ERROR
            )

            print(
                traceback.format_exc()
            )


    threading.Thread(
        target=_run,
        daemon=True
    ).start()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_kb(
    query: str
) -> List[Dict[str, Any]]:

    started = (
        time.perf_counter()
    )

    try:

        collection_count = (
            kb_col.count()
        )

    except Exception:

        collection_count = 0


    if collection_count <= 0:

        print(
            "[WARN] KB retrieval requested "
            "but collection is empty."
        )

        return []


    # Pull extra candidates so one large document does not
    # automatically monopolize every final retrieval slot.

    candidate_count = min(
        collection_count,
        max(
            MAX_KB_HITS * 3,
            MAX_KB_HITS
        )
    )


    result = kb_col.query(
        query_texts=[
            query
        ],
        n_results=
            candidate_count
    )


    docs = (
        result
        .get(
            "documents",
            [[]]
        )[0]
    )

    metas = (
        result
        .get(
            "metadatas",
            [[]]
        )[0]
    )

    distances = (
        result
        .get(
            "distances",
            [[]]
        )[0]
    )


    candidates = []

    for index, (
        doc,
        meta
    ) in enumerate(
        zip(
            docs,
            metas
        )
    ):

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        candidates.append(
            {
                "text":
                    doc,

                "meta":
                    meta or {},

                "distance":
                    distance,
            }
        )


    hits = []
    source_counts = {}
    context_chars = 0


    for candidate in candidates:

        source = (
            candidate["meta"]
            .get(
                "source",
                "Unknown source"
            )
        )

        used_from_source = (
            source_counts.get(
                source,
                0
            )
        )

        if (
            used_from_source
            >= MAX_HITS_PER_SOURCE
        ):
            continue


        text = (
            candidate["text"]
            or ""
        )[
            :MAX_CHARS_PER_CHUNK
        ]


        if not text.strip():
            continue


        remaining = (
            MAX_CONTEXT_CHARS
            - context_chars
        )

        if remaining <= 0:
            break


        if len(text) > remaining:
            text = text[:remaining]


        hit = {
            "text":
                text,

            "meta":
                candidate["meta"],

            "distance":
                candidate["distance"],
        }


        hits.append(
            hit
        )

        source_counts[source] = (
            used_from_source
            + 1
        )

        context_chars += len(
            text
        )


        if (
            len(hits)
            >= MAX_KB_HITS
        ):
            break


    print(
        "[RETRIEVAL] "
        + " | ".join(
            (
                f"{hit['meta'].get('source')} "
                f"[{hit['meta'].get('source_type')}] "
                f"chunk={hit['meta'].get('chunk')}"
            )
            for hit in hits
        )
    )


    _log_timing(
        "KB retrieval",
        started
    )


    return hits


def _format_kb_context(
    kb_hits: List[Dict[str, Any]]
) -> str:

    if not kb_hits:
        return (
            "No directly relevant internal "
            "research was retrieved."
        )


    blocks = []

    for hit in kb_hits:

        meta = (
            hit.get("meta")
            or {}
        )

        source = (
            meta.get(
                "source",
                "Unknown source"
            )
        )

        source_type = (
            meta.get(
                "source_type",
                "knowledge_base"
            )
        )

        chunk = meta.get(
            "chunk"
        )

        blocks.append(
            (
                f"[Source: {source} | "
                f"Type: {source_type} | "
                f"Chunk: {chunk}]\n"
                f"{hit.get('text', '')}"
            )
        )


    return "\n\n".join(
        blocks
    )


# ============================================================
# ANSWER CACHE
# ============================================================

ANSWER_CACHE: Dict[
    str,
    Dict[str, Any]
] = {}


def _cache_result(
    key: str,
    result: Dict[str, Any]
):

    if (
        len(ANSWER_CACHE)
        >= ANSWER_CACHE_MAX
    ):

        oldest_key = next(
            iter(
                ANSWER_CACHE
            )
        )

        ANSWER_CACHE.pop(
            oldest_key,
            None
        )


    ANSWER_CACHE[key] = result


# ============================================================
# RETRIEVAL QUERY LOGIC
# ============================================================

_CONTEXT_DEPENDENT_PATTERNS = [
    r"\bthis\b",
    r"\bthat\b",
    r"\bthese\b",
    r"\bthose\b",
    r"\bit\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\btheir\b",
    r"\bwhich one\b",
    r"\bwhich of\b",
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\btell me more\b",
    r"\bgo deeper\b",
    r"\bexpand on\b",
    r"\bwhy\b",
]


def _needs_followup_rewrite(
    messages: List[Dict[str, str]]
) -> bool:
    """
    Decide whether a follow-up is ambiguous enough
    to justify a separate LLM rewrite call.

    First-turn questions NEVER need rewriting.
    """

    user_messages = [
        m
        for m in messages
        if m.get("role") == "user"
        and m.get("content")
    ]


    if len(user_messages) <= 1:
        return False


    latest = (
        user_messages[-1]["content"]
        .strip()
        .lower()
    )


    # Very short follow-ups are often contextual.
    if len(latest.split()) <= 7:
        return True


    for pattern in _CONTEXT_DEPENDENT_PATTERNS:

        if re.search(
            pattern,
            latest
        ):
            return True


    return False


def _rewrite_retrieval_query(
    messages: List[Dict[str, str]],
    latest_question: str
) -> str:

    _ensure_client()

    started = (
        time.perf_counter()
    )


    recent_context = "\n".join(
        f"{m['role']}: "
        f"{m['content']}"
        for m
        in messages[-6:]
    )


    prompt = f"""
Rewrite the user's latest question as a concise, standalone search query
for retrieving relevant HR research.

Use the conversation only to resolve references such as "that", "these",
"which one", or "tell me more".

Do not answer the question.
Return only the rewritten search query.

Conversation:
{recent_context}

Latest question:
{latest_question}
"""


    try:

        response = (
            client
            .chat.completions
            .create(
                model=CHAT_MODEL,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            (
                                "Rewrite ambiguous "
                                "follow-up questions "
                                "into concise standalone "
                                "retrieval queries."
                            ),
                    },
                    {
                        "role":
                            "user",

                        "content":
                            prompt,
                    },
                ],

                temperature=0,

                max_tokens=100,
            )
        )


        rewritten = (
            response
            .choices[0]
            .message
            .content
            or latest_question
        ).strip()


        return (
            rewritten
            or latest_question
        )


    except Exception as exc:

        print(
            "[WARN] Retrieval rewrite failed:",
            exc
        )

        return latest_question


    finally:

        _log_timing(
            "Follow-up query rewrite",
            started
        )


# ============================================================
# RESULT HELPER
# ============================================================

def _public_kb_hits(
    kb_hits: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    return [
        {
            "source":
                hit["meta"]
                .get(
                    "source"
                ),

            "source_type":
                hit["meta"]
                .get(
                    "source_type"
                ),

            "chunk":
                hit["meta"]
                .get(
                    "chunk"
                ),

            "text":
                hit["text"][:700],
        }

        for hit
        in kb_hits
    ]


# ============================================================
# SINGLE-TURN ANSWER
# ============================================================

def generate_answer(
    question: str
) -> Dict[str, Any]:

    _ensure_client()


    started_total = (
        time.perf_counter()
    )


    question = (
        question
        or ""
    ).strip()


    qkey = (
        question.lower()
    )


    if qkey in ANSWER_CACHE:

        print(
            "[CACHE] Answer hit"
        )

        return (
            ANSWER_CACHE[qkey]
        )


    kb_hits = retrieve_kb(
        question
    )


    kb_context = (
        _format_kb_context(
            kb_hits
        )
    )


    user_prompt = f"""
Question:
{question}

Relevant HRNXT / Executive Networks research context:
{kb_context}

Use the context when it genuinely helps answer the question.
Do not force weakly related material into the answer.
"""


    generation_started = (
        time.perf_counter()
    )


    response = (
        client
        .chat.completions
        .create(
            model=CHAT_MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        ASK_MIKE_SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],

            temperature=0.2,

            max_tokens=
                MAX_ANSWER_TOKENS,
        )
    )


    _log_timing(
        "Final answer generation",
        generation_started
    )


    answer_text = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()


    result = {
        "answer":
            answer_text,

        "kb_hits":
            _public_kb_hits(
                kb_hits
            ),

        # Reserved for the future trusted-source/web-search layer.
        "web_domain_snippets":
            [],

        "web_people_snippets":
            [],

        "g_tags":
            [],
    }


    _cache_result(
        qkey,
        result
    )


    _log_timing(
        "Total single-turn request",
        started_total
    )


    return result


# ============================================================
# MULTI-TURN ANSWER
# ============================================================

def generate_answer_from_messages(
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:

    _ensure_client()


    started_total = (
        time.perf_counter()
    )


    clean_messages = [
        {
            "role":
                m.get("role"),

            "content":
                str(
                    m.get("content")
                    or ""
                ).strip(),
        }

        for m
        in messages

        if (
            m.get("role")
            in (
                "user",
                "assistant"
            )
            and
            m.get("content")
        )
    ]


    if not clean_messages:

        return generate_answer(
            ""
        )


    latest_question = (
        clean_messages[-1]
        ["content"]
    )


    # --------------------------------------------------------
    # FIRST TURN FAST PATH
    # --------------------------------------------------------

    user_message_count = sum(
        1
        for m
        in clean_messages
        if m["role"] == "user"
    )


    if user_message_count == 1:

        print(
            "[FAST PATH] "
            "First turn: skipping retrieval rewrite"
        )

        return generate_answer(
            latest_question
        )


    # --------------------------------------------------------
    # FOLLOW-UP RETRIEVAL QUERY
    # --------------------------------------------------------

    if _needs_followup_rewrite(
        clean_messages
    ):

        print(
            "[FOLLOW-UP] "
            "Context-dependent question: "
            "using retrieval rewrite"
        )

        retrieval_query = (
            _rewrite_retrieval_query(
                clean_messages,
                latest_question
            )
        )

    else:

        print(
            "[FOLLOW-UP] "
            "Question appears standalone: "
            "skipping retrieval rewrite"
        )

        retrieval_query = (
            latest_question
        )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    kb_hits = retrieve_kb(
        retrieval_query
    )


    kb_context = (
        _format_kb_context(
            kb_hits
        )
    )


    # Keep a reasonable amount of history.
    conversation_messages = (
        clean_messages[-8:]
    )


    openai_messages = [
        {
            "role":
                "system",

            "content":
                (
                    ASK_MIKE_SYSTEM_PROMPT
                    +
                    "\n\n"
                    "This is a continuing conversation. "
                    "Use prior messages to understand "
                    "the user's context and avoid "
                    "repeating information unnecessarily."
                ),
        },
        {
            "role":
                "user",

            "content":
                (
                    "Relevant HRNXT / Executive Networks "
                    "research context for the latest question:\n"
                    f"{kb_context}\n\n"
                    "Use the context when it genuinely helps. "
                    "Do not force weakly related material "
                    "into the answer."
                ),
        },
    ] + conversation_messages


    generation_started = (
        time.perf_counter()
    )


    response = (
        client
        .chat.completions
        .create(
            model=CHAT_MODEL,

            messages=
                openai_messages,

            temperature=0.2,

            max_tokens=
                MAX_ANSWER_TOKENS,
        )
    )


    _log_timing(
        "Final follow-up generation",
        generation_started
    )


    answer_text = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()


    result = {
        "answer":
            answer_text,

        "kb_hits":
            _public_kb_hits(
                kb_hits
            ),

        "retrieval_query":
            retrieval_query,

        # Reserved for the future trusted-source/web-search layer.
        "web_domain_snippets":
            [],

        "web_people_snippets":
            [],

        "g_tags":
            [],
    }


    _log_timing(
        "Total follow-up request",
        started_total
    )


    return result
