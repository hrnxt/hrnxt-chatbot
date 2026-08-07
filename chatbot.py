# chatbot.py
# HRNXT Ask Mike
# Ask Mike V13: contextualized answering + answer-first direct research review
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
from datetime import datetime, timezone
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
# Use smaller, more evidence-level chunks so semantic retrieval can
# surface a specific study/finding rather than a broad multi-topic block.
# Retrieved chunks are then passed essentially whole to the answer model,
# subject to a modest overall context cap.

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
        "2600"
    )
)

MAX_CONTEXT_CHARS = int(
    os.environ.get(
        "MAX_CONTEXT_CHARS",
        "10000"
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
You are Ask Mike, an HR adviser for senior HR leaders.

Match the answer to the question actually asked.

For simple factual questions:
- Answer directly and concisely.
- Do not add strategy, tensions, frameworks, recommendations, pilots, surveys,
  governance, or next steps unless the user asks for them.
- Do not turn a basic fact into an executive-advisory answer.

For substantive HR, advisory, comparative, or action-oriented questions:
- Respond like a thoughtful, experienced peer to a CHRO or senior HR executive.
- Lead with the most useful judgment, not a generic summary of the topic.
- Be pragmatic, specific, commercially aware, and willing to take a point of view.
- When the question asks "which", "what first", "most important", or "what would you do",
  choose and defend a position rather than listing several equally weighted possibilities.
- When the question explicitly presents a tension ("worth it or overhyped", "centralize or localize",
  "AI or human judgment"), address both the case for and the strongest limitation before landing
  on your recommendation.
- Prefer two or three substantive ideas over a long framework.
- Explain the important tradeoff, boundary condition, execution risk, or
  organizational implication when it matters.
- Preserve useful nuance: enterprise standards and local execution can coexist;
  avoid forcing a whole HR capability into a purely centralized or decentralized box
  when the better answer is to split decision rights within that capability.
- Give a practical next move only when it genuinely advances the answer.
  A next step is optional, not mandatory.

Style:
- Use plain English and short paragraphs.
- Do not restate the user's question.
- Avoid generic consultant filler.
- Do not automatically create headings, bullets, debate questions, frameworks,
  pilots, surveys, metrics, governance structures, or next steps.
- Do not create artificial tension when the question has a straightforward answer.

Research use:
- Your primary job is to answer the user's question with sound executive judgment.
- Research is supporting evidence, not a script. Do not let retrieved material replace
  your own reasoning or pull the answer toward a merely interesting adjacent topic.
- Form the clearest answer you can to the user's actual question, then use a supplied
  research synthesis to sharpen, challenge, qualify, or make that judgment more specific
  when it genuinely adds value.
- If the synthesis does not improve the answer, leave it out.
- When the user asks for a choice, priority, or "most important" judgment, make one.
  Do not substitute a list of loosely related research themes.
- Prefer durable insights and implications over precise statistics from static research.
- Do not treat analogous evidence as direct proof.
- Never invent support or claim the research says something it does not.
- Material labeled thought_leadership may be a bibliography or map to ideas rather
  than the full underlying work; do not pretend you have read an underlying source
  unless its content is actually present.

Follow-ups:
- Build on the prior conversation.
- Avoid unnecessary repetition.
- If the user asks "which one", "what first", or "what would you do", make a choice
  unless the available context genuinely prevents one.
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
    max_tokens: int = 400
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

    print(
        "[INFO] Retrieval config: "
        "chunk_tokens=400 "
        f"max_hits={MAX_KB_HITS} "
        f"max_chars_per_chunk={MAX_CHARS_PER_CHUNK} "
        f"max_context_chars={MAX_CONTEXT_CHARS} "
        "mode=natural_semantic"
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
    # Because chunks are smaller (~400 tokens), each candidate should
    # represent a more specific study, finding, or idea.

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
# REQUEST ROUTING + RESEARCH SYNTHESIS
# ============================================================

_FACTUAL_PATTERNS = [
    r"^\s*how old\b",
    r"^\s*when (?:is|was|did|does|do|will|are|were)\b",
    r"^\s*who (?:is|was|are|were)\b",
    r"^\s*where (?:is|was|are|were)\b",
    r"^\s*what (?:is|are|was|were) (?:the )?(?:age|date|year|meaning|definition|difference between)\b",
    r"^\s*define\b",
    r"^\s*what does .+ mean\??\s*$",
]

_ADVISORY_OR_RESEARCH_PATTERNS = [
    r"\bshould\b",
    r"\bmost likely\b",
    r"\bwhat first\b",
    r"\bwhere should\b",
    r"\bhow should\b",
    r"\bwhat would you do\b",
    r"\bwhat do you recommend\b",
    r"\brecommend\b",
    r"\bstrategy\b",
    r"\bstrategic\b",
    r"\bprioriti[sz]e\b",
    r"\btrade-?off\b",
    r"\bgovernance\b",
    r"\boperating model\b",
    r"\broadmap\b",
    r"\bframework\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bworth pursuing\b",
    r"\boverhyped\b",
    r"\bexpect .+ to change\b",
]

_EVIDENCE_REQUEST_PATTERNS = [
    r"\bevidence\b",
    r"\bdata\b",
    r"\bstatistics?\b",
    r"\bstudies\b",
    r"\bresearch\b",
    r"\bsources?\b",
    r"\bcitations?\b",
    r"\baccording to\b",
]


def _current_date_string() -> str:
    """
    Supply the model with an explicit current date for time-sensitive
    factual questions. UTC avoids depending on Render's local timezone.
    """

    return (
        datetime.now(timezone.utc)
        .strftime("%B %-d, %Y")
    )


def _is_explicit_evidence_request(
    question: str
) -> bool:

    q = (
        question
        or ""
    ).lower()

    return any(
        re.search(pattern, q)
        for pattern
        in _EVIDENCE_REQUEST_PATTERNS
    )


def _request_needs_research(
    question: str
) -> bool:
    """
    Low-cost routing heuristic.

    Obvious factual questions skip retrieval entirely.
    Advisory, prioritization, comparative, and research-oriented
    questions use the Ask Mike research pipeline.

    Ambiguous questions default to the research path because Ask Mike's
    primary job is substantive HR guidance.
    """

    q = (
        question
        or ""
    ).strip().lower()


    if any(
        re.search(pattern, q)
        for pattern
        in _ADVISORY_OR_RESEARCH_PATTERNS
    ):
        return True


    if any(
        re.search(pattern, q)
        for pattern
        in _EVIDENCE_REQUEST_PATTERNS
    ):
        return True


    if any(
        re.search(pattern, q)
        for pattern
        in _FACTUAL_PATTERNS
    ):
        return False


    # Very short "what is X?" / "what are X?" questions are usually
    # definitional rather than requests for strategic advice.
    if (
        len(q.split()) <= 10
        and re.match(
            r"^\s*what (?:is|are)\b",
            q
        )
        and not re.search(
            r"\b(best|most|better|important|critical|effective)\b",
            q
        )
    ):
        return False


    return True


def _question_wants_thought_leadership(
    question: str
) -> bool:
    """
    Thinkers50 files are citation maps / thought-leadership references,
    not the default evidence base. Use them only when the user is
    explicitly asking about thinkers, authors, books, frameworks,
    sources, or who to read/follow.
    """

    q = (
        question
        or ""
    ).lower()

    patterns = [
        r"\bthinkers?50\b",
        r"\bthought leader",
        r"\bthought leadership\b",
        r"\bmanagement thinker",
        r"\bauthors?\b",
        r"\bbooks?\b",
        r"\barticles?\b",
        r"\bframeworks?\b",
        r"\bwho should i read\b",
        r"\bwho should we read\b",
        r"\bwho should i follow\b",
        r"\bwho should we follow\b",
        r"\bwho writes about\b",
        r"\bwhose work\b",
        r"\bwhat should i read\b",
        r"\brecommended reading\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern
        in patterns
    )


def _filter_hits_for_question(
    question: str,
    kb_hits: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Keep substantive research layers for ordinary Ask Mike questions.
    Thought-leadership citation files only participate when the user's
    question actually calls for them.
    """

    if _question_wants_thought_leadership(
        question
    ):
        return kb_hits

    filtered = [
        hit
        for hit
        in kb_hits
        if (
            hit.get("meta", {})
            .get("source_type")
            != "thought_leadership"
        )
    ]

    if len(filtered) != len(kb_hits):
        print(
            f"[RETRIEVAL FILTER] removed_thought_leadership="
            f"{len(kb_hits) - len(filtered)}"
        )

    return filtered


def _synthesize_research(
    question: str,
    kb_hits: List[Dict[str, Any]]
) -> str:
    """
    Convert raw retrieved excerpts into a compact evidence brief before
    final answer generation. This gives gpt-4o-mini a much easier task:
    identify what the evidence actually changes before writing advice.
    """

    if not kb_hits:
        return ""


    _ensure_client()


    kb_context = (
        _format_kb_context(
            kb_hits
        )
    )


    allow_specifics = (
        _is_explicit_evidence_request(
            question
        )
    )


    if allow_specifics:

        specificity_instruction = (
            "The user explicitly asked for evidence/data/sources. "
            "You may preserve useful specific findings from the excerpts, "
            "but do not invent anything and note when evidence is indirect."
        )

    else:

        specificity_instruction = (
            "Do NOT repeat precise statistics, percentages, dates, forecasts, "
            "market-size figures, named laws, named regulations, citation names, "
            "or named studies. Translate them into durable qualitative implications "
            "instead unless the user explicitly requested evidence or sources."
        )


    prompt = f"""
User question:
{question}

Retrieved research excerpts:
{kb_context}

Create a compact supporting research brief for another model that will answer the user.

The final model will make its own executive judgment. Your job is only to surface
research that genuinely sharpens, challenges, qualifies, or makes that judgment
more specific. Do not choose the answer on the final model's behalf.

FIRST apply a hard relevance test:
- Does the retrieved material directly help answer the user's actual question?
- Does it materially change, sharpen, qualify, or constrain the answer?
- Is it more than merely interesting or topically adjacent?

If the answer is NO, return exactly:
NO_MATERIAL_RESEARCH

Otherwise use this compact structure, including only sections that are genuinely supported:

SUPPORT:
- 1 or 2 concise durable findings or principles the research supports.

CAUTION:
- 0 to 2 concise limitations, tradeoffs, risks, boundary conditions, or reasons
  not to overgeneralize the research.

JUDGMENT:
- What remains a leadership choice, context-dependent decision, or unresolved
  question rather than something the research itself determines.

Rules:
- Do not force all three sections if the evidence does not support them.
- SUPPORT should not merely restate the user's premise.
- CAUTION is especially important for questions that ask whether something is
  "worth it", "overhyped", "best", "most likely", "what first", or otherwise invite
  a tradeoff or prioritization judgment.
- JUDGMENT should preserve room for executive reasoning; do not make the final
  recommendation on the answering model's behalf.
- Answer relevance outranks novelty. Interesting adjacent research should be discarded.
- Respect singular questions. If the user asks for the single most important thing,
  synthesize around one central issue rather than listing several unrelated themes.
- Distinguish direct evidence from analogy.
- A finding from one occupation/function/use case does not prove the same effect elsewhere.
- When evidence is indirect, state the transferable principle rather than pretending
  the user's exact application was tested.
- For "where", "which", "most likely", or "what first" questions, identify the
  characteristics of the tasks/situations supported by the evidence before mapping
  them to HR applications.
- Exclude generic HR advice that is not actually sharpened by the excerpts.
- Do not recommend a pilot, governance framework, survey, or next step unless the
  evidence itself makes that recommendation materially important.
- {specificity_instruction}

Return only the brief bullets.
"""


    started = (
        time.perf_counter()
    )


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
                                "Extract the strongest decision-relevant "
                                "implications from retrieved HR research. "
                                "Be concise and evidence-disciplined."
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

                max_tokens=220,
            )
        )


        synthesis = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()


        print(
            f"[SYNTHESIS] chars={len(synthesis)} "
            f"evidence_request={allow_specifics}"
        )

        print(
            "[SYNTHESIS OUTPUT]\n"
            f"{synthesis}"
        )


        return synthesis


    except Exception as exc:

        print(
            "[WARN] Research synthesis failed:",
            exc
        )

        return ""


    finally:

        _log_timing(
            "Research synthesis",
            started
        )


def _generate_direct_answer(
    question: str,
    conversation_messages: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Fast path for straightforward factual/explanatory questions.
    No KB retrieval, no executive-advisory scaffolding.
    """

    _ensure_client()


    current_date = (
        _current_date_string()
    )


    direct_system = f"""
You are Ask Mike.

Current date: {current_date}.

Answer the user's straightforward factual or explanatory question directly.

- Give the answer in the first sentence.
- Be concise unless the user asks for detail.
- Do not add HR strategy, executive implications, tensions, recommendations,
  pilots, surveys, frameworks, governance, or next steps unless explicitly requested.
- For age/date questions, calculate relative to the current date supplied above.
- If a fact depends on an uncertain convention or definition, state the convention briefly.
"""


    messages = [
        {
            "role":
                "system",

            "content":
                direct_system,
        }
    ]


    if conversation_messages:

        messages += (
            conversation_messages[-8:]
        )

    else:

        messages.append(
            {
                "role":
                    "user",

                "content":
                    question,
            }
        )


    started = (
        time.perf_counter()
    )


    response = (
        client
        .chat.completions
        .create(
            model=CHAT_MODEL,

            messages=
                messages,

            temperature=0.1,

            max_tokens=250,
        )
    )


    _log_timing(
        "Direct answer generation",
        started
    )


    answer_text = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()


    return {
        "answer":
            answer_text,

        "kb_hits":
            [],

        "route":
            "direct",

        "web_domain_snippets":
            [],

        "web_people_snippets":
            [],

        "g_tags":
            [],
    }



def _build_contextual_retrieval_query(
    messages: List[Dict[str, str]],
    latest_question: str
) -> str:
    """
    Build a deterministic follow-up retrieval query from recent USER turns.

    This avoids an extra model call and prevents the retrieval rewrite from
    accidentally changing the subject of the conversation.

    We intentionally favor the immediately preceding user question plus the
    latest follow-up. If the previous user turn is itself very short/contextual,
    include one additional earlier user turn when available.
    """

    user_messages = [
        (
            m.get("content")
            or ""
        ).strip()
        for m
        in messages
        if (
            m.get("role") == "user"
            and m.get("content")
        )
    ]


    if len(user_messages) <= 1:
        return (
            latest_question
            or ""
        ).strip()


    latest = (
        latest_question
        or user_messages[-1]
    ).strip()

    previous = (
        user_messages[-2]
    ).strip()


    context_parts = []


    # If the prior user message is itself very short, include one
    # additional earlier user message to preserve the underlying topic.
    if (
        len(previous.split()) <= 8
        and len(user_messages) >= 3
    ):
        context_parts.append(
            user_messages[-3]
        )


    context_parts.append(
        previous
    )

    context_parts.append(
        latest
    )


    query = " | ".join(
        part
        for part
        in context_parts
        if part
    )


    # Keep embeddings focused if a conversation becomes unusually long.
    return query[:1200]


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


def _generate_independent_advisory_draft(
    question: str,
    conversation_messages: List[Dict[str, str]] = None
) -> str:
    """
    Generate Ask Mike's judgment before showing the model retrieved research.
    This preserves the model's independent reasoning and prevents retrieval
    from anchoring the initial answer.
    """

    _ensure_client()

    messages = [
        {
            "role": "system",
            "content": ASK_MIKE_SYSTEM_PROMPT,
        }
    ]

    if conversation_messages:
        messages += conversation_messages[-8:]
    else:
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

    started = time.perf_counter()

    response = (
        client
        .chat.completions
        .create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    )

    _log_timing(
        "Independent advisory draft",
        started
    )

    draft = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()

    print(
        "[INDEPENDENT DRAFT]\n"
        f"{draft}"
    )

    return draft


def _revise_draft_with_research(
    question: str,
    draft: str,
    research_brief: str
) -> str:
    """
    Research is introduced only AFTER Ask Mike has formed an independent answer.
    Revise only when the research genuinely improves that answer.
    """

    if (
        not research_brief
        or research_brief == "NO_MATERIAL_RESEARCH"
    ):
        print(
            "[RESEARCH REVISION] "
            "No material research; keeping independent draft"
        )
        return draft

    _ensure_client()

    prompt = f"""
User question:
{question}

Ask Mike's independent draft:
{draft}

Supporting research brief:
{research_brief}

Review the independent draft against the research brief.

Keep the independent draft unless the research reveals something that materially
improves it. Research is not an answer key.

Revise only to:
- correct or qualify an unsupported assumption,
- add an important boundary condition or counterargument,
- sharpen a recommendation,
- or add a durable insight that materially changes what the executive should conclude.

Do NOT:
- replace a strong judgment merely because the research discusses another topic,
- make the answer more generic,
- add research terminology for its own sake,
- add named studies, citations, laws, regulations, or precise static statistics
  unless the user explicitly requested evidence,
- add length unless the added material genuinely improves the answer.

For questions presenting a tension, preserve and answer that tension.
For "which", "what first", "most important", or "what would you do" questions,
make a clear choice.

Return only the final answer to the user.
"""

    started = time.perf_counter()

    response = (
        client
        .chat.completions
        .create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are editing an already-formed executive HR judgment. "
                        "Preserve its strengths. Use research conservatively and only "
                        "when it materially improves the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    )

    _log_timing(
        "Research revision",
        started
    )

    final_answer = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()

    print(
        "[REVISED ANSWER]\n"
        f"{final_answer}"
    )

    return final_answer



def _revise_draft_with_retrieved_context(
    answering_question: str,
    draft: str,
    kb_hits: List[Dict[str, Any]]
) -> str:
    """
    One conservative research-review pass using retrieved excerpts directly.

    The answering model has already formed its judgment. The retrieved material
    is allowed to correct, qualify, or sharpen that judgment, but not redirect it.
    This removes the separate research-synthesis model call from the live path.
    """

    if not kb_hits:
        print(
            "[RESEARCH REVIEW] "
            "No retrieved research; keeping independent draft"
        )
        return draft

    research_context = _format_kb_context(
        kb_hits
    )

    if not research_context.strip():
        print(
            "[RESEARCH REVIEW] "
            "Empty research context; keeping independent draft"
        )
        return draft

    allow_specifics = (
        _question_requests_evidence(
            answering_question
        )
    )

    if allow_specifics:
        specificity_rule = (
            "The user explicitly asked for evidence/data/sources. "
            "You may use specific evidence that is actually present in the excerpts, "
            "but do not invent or overstate it."
        )
    else:
        specificity_rule = (
            "Do not add precise statistics, dates, forecasts, named studies, citation "
            "names, named laws, or regulatory details. Convert useful research into "
            "durable qualitative implications."
        )

    prompt = f"""
Question to answer:
{answering_question}

Ask Mike's independent draft:
{draft}

Retrieved research excerpts:
{research_context}

Review the independent draft against the retrieved research.

IMPORTANT:
- The independent draft is the starting judgment.
- Research is supporting material, not an answer key.
- Keep the draft unchanged if the excerpts do not materially improve it.
- Ignore research that is merely adjacent, interesting, or less responsive to the
  actual question than the draft.
- Do not introduce a new topic simply because it appears in retrieval.
- For a question presenting a tension, preserve both the strongest case and the
  strongest limitation before landing on a judgment.
- For "which", "what first", "most important", or "what would you do" questions,
  make a clear choice.
- Revise only to correct an unsupported assumption, add an important counterargument
  or boundary condition, sharpen a recommendation, or add a durable insight that
  materially changes what the executive should conclude.
- Preserve useful specificity and executive judgment from the draft.
- Do not make the answer longer unless the added material genuinely improves it.
- {specificity_rule}

Return only the final answer to the user.
"""

    started = time.perf_counter()

    response = (
        client
        .chat.completions
        .create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are conservatively reviewing an already-formed executive "
                        "HR answer against retrieved internal research. Preserve the "
                        "answer's independent judgment unless the research materially "
                        "improves it."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    )

    _log_timing(
        "Research review",
        started
    )

    final_answer = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()

    print(
        "[RESEARCH-REVIEWED ANSWER]\n"
        f"{final_answer}"
    )

    return final_answer


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


    needs_research = (
        _request_needs_research(
            question
        )
    )


    print(
        f"[ROUTE] "
        f"{'research' if needs_research else 'direct'}"
    )


    # Include route + current date in cache key so a factual answer
    # involving "now" cannot silently survive across dates.
    cache_date = (
        _current_date_string()
        if not needs_research
        else "research"
    )

    qkey = (
        f"{cache_date}:"
        f"{question.lower()}"
    )


    if qkey in ANSWER_CACHE:

        print(
            "[CACHE] Answer hit"
        )

        return (
            ANSWER_CACHE[qkey]
        )


    if not needs_research:

        result = (
            _generate_direct_answer(
                question
            )
        )

        _cache_result(
            qkey,
            result
        )

        _log_timing(
            "Total direct request",
            started_total
        )

        return result


    # --------------------------------------------------------
    # RESEARCH PATH
    # --------------------------------------------------------

    kb_hits = retrieve_kb(
        question
    )

    kb_hits = _filter_hits_for_question(
        question,
        kb_hits
    )


    source_types = {}

    for hit in kb_hits:

        source_type = (
            hit.get("meta", {})
            .get(
                "source_type",
                "knowledge_base"
            )
        )

        source_types[source_type] = (
            source_types.get(
                source_type,
                0
            )
            + 1
        )


    raw_context_chars = sum(
        len(
            hit.get(
                "text",
                ""
            )
        )
        for hit
        in kb_hits
    )


    print(
        f"[GROUNDING] hits={len(kb_hits)} "
        f"raw_context_chars={raw_context_chars} "
        f"source_types={source_types}"
    )


    # Form the executive judgment BEFORE exposing the answering model
    # to retrieved research.
    independent_draft = (
        _generate_independent_advisory_draft(
            question
        )
    )


    # V13: no separate synthesis call. Review the independent answer
    # directly against the retrieved excerpts.
    answer_text = (
        _revise_draft_with_retrieved_context(
            question,
            independent_draft,
            kb_hits
        )
    )

    research_brief = None


    result = {
        "answer":
            answer_text,

        "kb_hits":
            _public_kb_hits(
                kb_hits
            ),

        "research_synthesis":
            research_brief,

        "independent_draft":
            independent_draft,

        "route":
            "research",

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
        "Total research request",
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


    user_message_count = sum(
        1
        for m
        in clean_messages
        if m["role"] == "user"
    )


    if user_message_count == 1:

        print(
            "[FAST PATH] "
            "First turn: routing normally"
        )

        return generate_answer(
            latest_question
        )


    needs_research = (
        _request_needs_research(
            latest_question
        )
    )


    print(
        f"[ROUTE] followup_"
        f"{'research' if needs_research else 'direct'}"
    )


    if not needs_research:

        result = (
            _generate_direct_answer(
                latest_question,
                clean_messages
            )
        )

        _log_timing(
            "Total direct follow-up",
            started_total
        )

        return result


    # --------------------------------------------------------
    # FOLLOW-UP RETRIEVAL QUERY
    # --------------------------------------------------------

    print(
        "[FOLLOW-UP] "
        "Building deterministic contextual "
        "retrieval query"
    )

    retrieval_query = (
        _build_contextual_retrieval_query(
            clean_messages,
            latest_question
        )
    )

    print(
        "[FOLLOW-UP QUERY] "
        f"{retrieval_query}"
    )


    kb_hits = retrieve_kb(
        retrieval_query
    )

    kb_hits = _filter_hits_for_question(
        latest_question,
        kb_hits
    )


    source_types = {}

    for hit in kb_hits:

        source_type = (
            hit.get("meta", {})
            .get(
                "source_type",
                "knowledge_base"
            )
        )

        source_types[source_type] = (
            source_types.get(
                source_type,
                0
            )
            + 1
        )


    raw_context_chars = sum(
        len(
            hit.get(
                "text",
                ""
            )
        )
        for hit
        in kb_hits
    )


    print(
        f"[GROUNDING] followup_hits={len(kb_hits)} "
        f"raw_context_chars={raw_context_chars} "
        f"source_types={source_types}"
    )


    # V13: the same deterministic contextual question used for retrieval
    # is also used for answer generation. This prevents a short follow-up
    # from losing the subject of the prior exchange.
    answering_question = (
        retrieval_query
    )

    print(
        "[ANSWERING QUESTION] "
        f"{answering_question}"
    )


    independent_draft = (
        _generate_independent_advisory_draft(
            answering_question
        )
    )


    answer_text = (
        _revise_draft_with_retrieved_context(
            answering_question,
            independent_draft,
            kb_hits
        )
    )

    research_brief = None


    result = {
        "answer":
            answer_text,

        "kb_hits":
            _public_kb_hits(
                kb_hits
            ),

        "retrieval_query":
            retrieval_query,

        "research_synthesis":
            research_brief,

        "independent_draft":
            independent_draft,

        "route":
            "research",

        "web_domain_snippets":
            [],

        "web_people_snippets":
            [],

        "g_tags":
            [],
    }


    _log_timing(
        "Total research follow-up",
        started_total
    )


    return result
