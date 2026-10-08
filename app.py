import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from rag_engine import process_files
from vector_store import get_embeddings

from hybrid_retriever import (
    get_documents_from_vector_store,
    build_bm25,
    hybrid_search,
)

from reranker import rerank_documents


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Please add OPENAI_API_KEY to your .env file."
    )
    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DB_PATH = Path("vector_db")

SUPPORTED_TYPES = [
    "pdf",
    "docx",
    "txt",
]

SEMANTIC_K = 15
BM25_K = 15
HYBRID_K = 10
RERANK_K = 5


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "documents": [],
    "messages": [],
    "chunk_count": 0,
    "index_signature": None,
    "vector_store": None,
    "bm25": None,
    "rag_documents": [],
    "vector_db_saved": False,
    "vector_db_error": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# PREMIUM UI CSS
# ============================================================

st.html(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
);


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
.stApp {

    background: #080D0B !important;

    color: #F4F1E8 !important;
}


.stApp,
button,
input,
textarea,
[class*="css"] {

    font-family:
        "DM Sans",
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}


header {

    background:
        transparent !important;
}


#MainMenu {

    visibility:
        hidden;
}


footer {

    visibility:
        hidden;
}


.block-container {

    max-width:
        1450px;

    padding-top:
        1rem;

    padding-bottom:
        8rem;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #0B120F 0%,
            #080D0B 100%
        ) !important;

    border-right:
        1px solid #1D2A23;
}


section[data-testid="stSidebar"] > div {

    padding-top:
        1.25rem;
}


/* ============================================================
   BRAND
   ============================================================ */

.brand {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;
}


.brand-mark {

    width:
        40px;

    height:
        40px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        12px;

    background:
        linear-gradient(
            145deg,
            #1F8A5B,
            #14583A
        );

    border:
        1px solid rgba(229,201,120,0.22);

    color:
        #E5C978;

    font-size:
        17px;

    font-weight:
        700;

    box-shadow:
        0 10px 30px
        rgba(31,138,91,0.18);
}


.brand-name {

    color:
        #F4F1E8;

    font-family:
        "Manrope",
        sans-serif !important;

    font-size:
        18px;

    font-weight:
        700;

    letter-spacing:
        -0.6px;
}


.brand-subtitle {

    margin-left:
        52px;

    margin-top:
        -3px;

    color:
        #657169;

    font-size:
        9px;
}


/* ============================================================
   STATUS
   ============================================================ */

.system-status {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    margin-top:
        27px;

    padding:
        10px 12px;

    border-radius:
        10px;

    background:
        rgba(255,255,255,0.015);

    border:
        1px solid #202D26;
}


.status-left {

    display:
        flex;

    align-items:
        center;

    gap:
        8px;

    color:
        #9BA49D;

    font-size:
        9px;

    font-weight:
        600;

    letter-spacing:
        0.5px;
}


.status-dot {

    width:
        6px;

    height:
        6px;

    border-radius:
        50%;

    background:
        #45D39A;

    box-shadow:
        0 0 9px
        rgba(69,211,154,0.65);
}


.status-version {

    color:
        #526057;

    font-size:
        8px;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

.side-label {

    margin-top:
        28px;

    margin-bottom:
        11px;

    color:
        #637068;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        1.7px;

    text-transform:
        uppercase;
}


.document-item {

    display:
        flex;

    align-items:
        center;

    gap:
        9px;

    padding:
        9px;

    margin-bottom:
        6px;

    border-radius:
        9px;

    background:
        #0F1713;

    border:
        1px solid #1D2A23;
}


.document-icon {

    width:
        30px;

    height:
        30px;

    flex-shrink:
        0;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        8px;

    background:
        #14221B;

    border:
        1px solid #294134;

    color:
        #E5C978;

    font-size:
        12px;
}


.document-info {

    min-width:
        0;
}


.document-name {

    overflow:
        hidden;

    white-space:
        nowrap;

    text-overflow:
        ellipsis;

    color:
        #D7DBD5;

    font-size:
        10px;

    font-weight:
        500;
}


.document-meta {

    margin-top:
        2px;

    color:
        #66736A;

    font-size:
        8px;
}


.sidebar-empty {

    padding:
        8px 2px;

    color:
        #536058;

    font-size:
        9px;

    line-height:
        1.5;
}


/* ============================================================
   WORKSPACE STATS
   ============================================================ */

.workspace-grid {

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        7px;
}


.workspace-stat {

    padding:
        13px;

    border-radius:
        10px;

    background:
        #0F1713;

    border:
        1px solid #1D2A23;
}


.workspace-value {

    color:
        #F1EEE4;

    font-family:
        "Manrope",
        sans-serif !important;

    font-size:
        18px;

    font-weight:
        700;
}


.workspace-label {

    margin-top:
        3px;

    color:
        #5E6A62;

    font-size:
        7px;

    letter-spacing:
        1px;

    text-transform:
        uppercase;
}


/* ============================================================
   TOPBAR
   ============================================================ */

.topbar {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding-bottom:
        15px;

    border-bottom:
        1px solid #18231E;
}


.topbar-left {

    color:
        #5E6962;

    font-size:
        9px;
}


.topbar-left span {

    color:
        #909990;
}


.topbar-right {

    display:
        flex;

    gap:
        7px;

    flex-wrap:
        wrap;

    justify-content:
        flex-end;
}


.top-chip {

    padding:
        6px 10px;

    border-radius:
        7px;

    background:
        #0D1511;

    border:
        1px solid #1D2A23;

    color:
        #68746C;

    font-size:
        7px;

    font-weight:
        500;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {

    padding:
        43px 0 28px;
}


.hero-kicker {

    display:
        flex;

    align-items:
        center;

    gap:
        8px;

    color:
        #C9A24D;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        2px;

    text-transform:
        uppercase;
}


.hero-kicker-dot {

    width:
        5px;

    height:
        5px;

    border-radius:
        50%;

    background:
        #C9A24D;
}


.hero-title {

    margin:
        12px 0 0;

    max-width:
        900px;

    color:
        #F4F1E8;

    font-family:
        "Manrope",
        sans-serif !important;

    font-size:
        50px;

    line-height:
        1.02;

    font-weight:
        700;

    letter-spacing:
        -2.8px;
}


.hero-gradient {

    background:
        linear-gradient(
            90deg,
            #D9B961 0%,
            #E5C978 40%,
            #B7D6A9 100%
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


.hero-description {

    max-width:
        700px;

    margin-top:
        16px;

    color:
        #89938C;

    font-size:
        12px;

    line-height:
        1.75;
}


.capabilities {

    display:
        flex;

    gap:
        7px;

    margin-top:
        18px;

    flex-wrap:
        wrap;
}


.capability {

    padding:
        7px 11px;

    border-radius:
        7px;

    background:
        #0D1511;

    border:
        1px solid #202D26;

    color:
        #738078;

    font-size:
        7px;
}


.capability:first-child {

    color:
        #BCA968;

    border-color:
        rgba(201,162,77,0.20);
}


/* ============================================================
   UPLOAD AREA
   ============================================================ */

.upload-workspace {

    margin-top:
        18px;

    padding:
        34px;

    border-radius:
        21px;

    background:

        radial-gradient(
            circle at 8% 50%,
            rgba(31,138,91,0.22),
            transparent 30%
        ),

        radial-gradient(
            circle at 88% 15%,
            rgba(201,162,77,0.09),
            transparent 25%
        ),

        linear-gradient(
            135deg,
            #10271B 0%,
            #101D16 48%,
            #0F1713 100%
        );

    border:
        1px solid #345642;

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,0.04),

        0 20px 60px
        rgba(0,0,0,0.30);
}


.upload-title {

    color:
        #F4F1E8;

    font-family:
        "Manrope",
        sans-serif !important;

    font-size:
        21px;

    font-weight:
        700;
}


.upload-subtitle {

    margin-top:
        7px;

    color:
        #89958D;

    font-size:
        11px;

    line-height:
        1.6;
}


.upload-formats {

    margin-top:
        12px;

    color:
        #C9A24D;

    font-size:
        8px;

    font-weight:
        600;

    letter-spacing:
        0.8px;
}


.upload-formats span {

    margin:
        0 6px;

    color:
        #52675A;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

div[data-testid="stFileUploader"] {

    margin-top:
        20px;

    padding:
        14px;

    border-radius:
        14px;

    background:
        rgba(8,13,11,0.40);

    border:
        1px dashed #557262;
}


div[data-testid="stFileUploaderDropzone"] {

    background:
        transparent !important;

    border:
        none !important;
}


div[data-testid="stFileUploaderDropzoneInstructions"] {

    color:
        #B8C2BA !important;

    opacity:
        1 !important;
}


div[data-testid="stFileUploaderDropzoneInstructions"] span {

    color:
        #B8C2BA !important;
}


div[data-testid="stFileUploaderDropzoneInstructions"] small {

    color:
        #7C8A81 !important;
}


div[data-testid="stFileUploaderDropzone"] button {

    background:
        linear-gradient(
            145deg,
            #C9A24D,
            #A98235
        ) !important;

    color:
        #10140F !important;

    border:
        none !important;

    border-radius:
        9px !important;

    font-weight:
        600 !important;
}


div[data-testid="stFileUploaderFile"] {

    background:
        #111C16 !important;

    border:
        1px solid #294134 !important;

    border-radius:
        9px !important;
}


div[data-testid="stFileUploaderFile"] span {

    color:
        #D9DED9 !important;
}


/* ============================================================
   READY CARD
   ============================================================ */

.ready-card {

    margin-top:
        18px;

    padding:
        25px;

    border-radius:
        18px;

    background:

        radial-gradient(
            circle at 7% 50%,
            rgba(69,211,154,0.10),
            transparent 28%
        ),

        #0E1914;

    border:
        1px solid #28503C;
}


.ready-row {

    display:
        flex;

    align-items:
        center;

    gap:
        16px;
}


.ready-icon {

    width:
        52px;

    height:
        52px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        15px;

    background:
        rgba(69,211,154,0.07);

    border:
        1px solid rgba(69,211,154,0.20);

    color:
        #5DD69A;

    font-size:
        19px;
}


.ready-title {

    color:
        #F4F1E8;

    font-family:
        "Manrope",
        sans-serif !important;

    font-size:
        18px;

    font-weight:
        700;
}


.ready-subtitle {

    margin-top:
        4px;

    color:
        #7C8981;

    font-size:
        10px;
}


/* ============================================================
   EXPLORE
   ============================================================ */

.try-label {

    margin-top:
        32px;

    margin-bottom:
        12px;

    color:
        #667169;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        1.7px;

    text-transform:
        uppercase;
}


.prompt-card {

    min-height:
        112px;

    padding:
        20px;

    border-radius:
        14px;

    background:
        linear-gradient(
            145deg,
            #101914,
            #0D1511
        );

    border:
        1px solid #24352C;
}


.prompt-icon {

    color:
        #D3B45F;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        0.8px;
}


.prompt-title {

    margin-top:
        10px;

    color:
        #C1C9C2;

    font-size:
        11px;

    line-height:
        1.55;
}


/* ============================================================
   CHAT MESSAGES
   ============================================================ */

div[data-testid="stChatMessage"] {

    background:
        transparent !important;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

div[data-testid="stChatInput"],
section[data-testid="stChatInput"] {

    background:
        #0E1712 !important;

    border:
        1px solid #4A6655 !important;

    border-radius:
        14px !important;

    box-shadow:
        0 18px 50px
        rgba(0,0,0,0.45) !important;

    color:
        #FFFFFF !important;
}


/* BaseWeb */

div[data-testid="stChatInput"]
[data-baseweb="textarea"] {

    background:
        #0E1712 !important;

    color:
        #FFFFFF !important;

    border:
        none !important;
}


div[data-testid="stChatInput"]
[data-baseweb="base-input"] {

    background:
        #0E1712 !important;

    color:
        #FFFFFF !important;

    border:
        none !important;
}


/* Actual textarea */

div[data-testid="stChatInput"] textarea,
section[data-testid="stChatInput"] textarea {

    background:
        #0E1712 !important;

    color:
        #FFFFFF !important;

    -webkit-text-fill-color:
        #FFFFFF !important;

    caret-color:
        #E5C978 !important;

    font-family:
        "DM Sans",
        sans-serif !important;

    font-size:
        13px !important;

    font-weight:
        500 !important;

    opacity:
        1 !important;

    border:
        none !important;

    outline:
        none !important;

    box-shadow:
        none !important;
}


/* Placeholder */

div[data-testid="stChatInput"]
textarea::placeholder,
section[data-testid="stChatInput"]
textarea::placeholder {

    color:
        #7F8B83 !important;

    -webkit-text-fill-color:
        #7F8B83 !important;

    opacity:
        1 !important;
}


/* Focus */

div[data-testid="stChatInput"]
textarea:focus,
section[data-testid="stChatInput"]
textarea:focus {

    background:
        #0E1712 !important;

    color:
        #FFFFFF !important;

    -webkit-text-fill-color:
        #FFFFFF !important;

    caret-color:
        #E5C978 !important;

    outline:
        none !important;

    box-shadow:
        none !important;
}


/* Input fallback */

div[data-testid="stChatInput"] input,
div[data-testid="stChatInput"] textarea {

    background-color:
        #0E1712 !important;

    color:
        #FFFFFF !important;

    -webkit-text-fill-color:
        #FFFFFF !important;

    opacity:
        1 !important;
}


/* Send button */

div[data-testid="stChatInput"] button {

    background:
        linear-gradient(
            145deg,
            #C9A24D,
            #A98535
        ) !important;

    color:
        #10140F !important;

    border:
        none !important;

    border-radius:
        10px !important;
}


div[data-testid="stChatInput"] button svg {

    color:
        #10140F !important;

    fill:
        #10140F !important;
}


/* Bottom fixed chat area */

[data-testid="stBottomBlockContainer"] {

    background:
        linear-gradient(
            180deg,
            rgba(8,13,11,0) 0%,
            #080D0B 35%
        ) !important;
}


[data-testid="stBottomBlockContainer"] > div {

    background:
        transparent !important;
}


/* Final textarea fallback */

textarea {

    color:
        #FFFFFF !important;

    -webkit-text-fill-color:
        #FFFFFF !important;

    caret-color:
        #E5C978 !important;
}


/* ============================================================
   SOURCES
   ============================================================ */

.source-wrapper {

    margin-top:
        14px;

    padding:
        14px;

    border-radius:
        11px;

    background:
        #101813;

    border:
        1px solid #2A392F;
}


.source-heading {

    color:
        #C9A24D;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        1.3px;

    text-transform:
        uppercase;

    margin-bottom:
        8px;
}


.source-item {

    margin-top:
        7px;

    padding:
        10px;

    border-radius:
        8px;

    background:
        rgba(255,255,255,0.018);

    border:
        1px solid #1E2B24;
}


.source-file {

    color:
        #D4D9D4;

    font-size:
        9px;

    font-weight:
        600;
}


.source-page {

    margin-top:
        4px;

    color:
        #D3B45F;

    font-size:
        8px;

    font-weight:
        600;
}


.source-preview {

    margin-top:
        6px;

    color:
        #68766D;

    font-size:
        8px;

    line-height:
        1.5;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .hero-title {

        font-size:
            38px;

        letter-spacing:
            -2px;
    }

    .topbar-right {

        display:
            none;
    }

}

</style>
"""
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_signature(files):
    """
    Creates a unique signature for uploaded files.
    """

    return tuple(
        (
            file.name,
            len(file.getvalue()),
        )
        for file in files
    )


def extract_document(result):
    """
    Supports different result structures returned by
    hybrid_retriever / reranker.
    """

    if result is None:
        return None

    # LangChain Document
    if hasattr(result, "page_content"):
        return result

    # Dictionary result
    if isinstance(result, dict):

        for key in (
            "document",
            "doc",
            "source_document",
        ):

            if result.get(key) is not None:
                return result[key]

    return None


def result_content(result):

    document = extract_document(result)

    if document is None:
        return ""

    return getattr(
        document,
        "page_content",
        "",
    )


def result_metadata(result):

    document = extract_document(result)

    if document is None:
        return {}

    return (
        getattr(
            document,
            "metadata",
            {},
        )
        or {}
    )


# ============================================================
# SAVE VECTOR DATABASE SAFELY
# ============================================================

def save_vector_database(vector_store):
    """
    Saves FAISS to vector_db.

    IMPORTANT:
    We never delete vector_db.

    If Windows/OneDrive temporarily locks the folder,
    the application continues using the FAISS object in memory.
    """

    VECTOR_DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        vector_store.save_local(
            str(VECTOR_DB_PATH)
        )

        return True, None

    except PermissionError as exc:

        return (
            False,
            (
                "Windows denied access to vector_db. "
                "The knowledge base is still available "
                "in memory for this Streamlit session. "
                "Close other Python/Streamlit processes "
                "or OneDrive locks and restart the app "
                "to persist the database."
            ),
        )

    except Exception as exc:

        return (
            False,
            str(exc),
        )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base(uploaded_files):

    if not uploaded_files:

        raise ValueError(
            "Please upload at least one document."
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We pass Streamlit UploadedFile objects DIRECTLY.
    #
    # We do NOT:
    #
    # - scan the project folder
    # - read s.pdf
    # - create paths
    # - convert files to strings
    #
    # Therefore only uploaded files are indexed.
    # --------------------------------------------------------

    st.write(
        "Extracting text from uploaded documents..."
    )

    chunks = process_files(
        uploaded_files
    )

    if not chunks:

        raise ValueError(
            "No readable text was found "
            "in the uploaded documents."
        )

    st.write(
        f"Created {len(chunks):,} document chunks."
    )

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    st.write(
        "Generating embeddings and building FAISS..."
    )

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # --------------------------------------------------------
    # SAVE FAISS
    # --------------------------------------------------------

    saved, save_error = save_vector_database(
        vector_store
    )

    st.session_state.vector_db_saved = saved

    st.session_state.vector_db_error = (
        save_error
    )

    if saved:

        st.write(
            "FAISS vector database saved to vector_db."
        )

    else:

        st.warning(
            save_error
        )

    # --------------------------------------------------------
    # GET DOCUMENTS DIRECTLY FROM CURRENT FAISS
    # --------------------------------------------------------

    documents = (
        get_documents_from_vector_store(
            vector_store
        )
    )

    if not documents:

        raise ValueError(
            "FAISS was created, but no documents "
            "were found in the vector store."
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    st.write(
        "Building BM25 keyword index..."
    )

    bm25 = build_bm25(
        documents
    )

    return (
        vector_store,
        bm25,
        documents,
        len(chunks),
    )


# ============================================================
# RERANK
# ============================================================

def perform_reranking(
    query,
    results,
    top_k=5,
):

    if not results:
        return []

    try:

        reranked = rerank_documents(
            query=query,
            results=results,
            top_k=top_k,
        )

        if reranked:
            return reranked

    except Exception as exc:

        st.warning(
            "Reranking was unavailable. "
            "Using hybrid retrieval results."
        )

    return results[:top_k]


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_grounded_answer(
    query,
    results,
):

    if not results:

        return {
            "answer": (
                "I couldn't find relevant information "
                "in the uploaded documents."
            ),
            "sources": [],
        }

    context_blocks = []

    sources = []

    seen_sources = set()

    for index, result in enumerate(results):

        content = result_content(
            result
        )

        metadata = result_metadata(
            result
        )

        if not content:
            continue

        source = metadata.get(
            "source",
            metadata.get(
                "file_name",
                "Unknown document",
            ),
        )

        page = metadata.get(
            "page",
            metadata.get(
                "page_number",
                None,
            ),
        )

        chunk_id = metadata.get(
            "chunk_id",
            index,
        )

        context_blocks.append(
            f"""
SOURCE {index + 1}

File: {source}
Page: {page if page is not None else "N/A"}
Chunk: {chunk_id}

{content}
"""
        )

        source_key = (
            str(source),
            str(page),
        )

        if source_key not in seen_sources:

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "content": content,
                    "chunk_id": chunk_id,
                }
            )

            seen_sources.add(
                source_key
            )

    context = "\n\n".join(
        context_blocks
    )

    prompt = f"""
You are DocuMind AI, a grounded document
question-answering assistant.

Answer the user's question ONLY using
the supplied document context.

USER QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

RULES:

1. Use only information contained in the supplied context.
2. Do not invent facts.
3. Do not use outside knowledge.
4. If the answer cannot be found in the context, say:
   "I couldn't find enough information in the uploaded documents."
5. Give a clear and concise answer.
6. Preserve technical terminology from the source.
7. Do not mention FAISS, BM25, embeddings, reranking,
   or internal retrieval processes.
8. Do not fabricate page numbers.
"""

    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
    )

    response = model.invoke(
        prompt
    )

    answer = getattr(
        response,
        "content",
        str(response),
    )

    return {
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# SOURCE DISPLAY
# ============================================================

def render_sources(sources):

    if not sources:
        return

    source_html = """
    <div class="source-wrapper">

        <div class="source-heading">
            Sources · Page-Level Citations
        </div>
    """

    for source in sources[:5]:

        file_name = html.escape(
            str(
                source.get(
                    "source",
                    "Unknown document",
                )
            )
        )

        page = source.get(
            "page"
        )

        content = source.get(
            "content",
            "",
        )

        preview = " ".join(
            str(content).split()
        )

        if len(preview) > 220:

            preview = (
                preview[:220]
                + "..."
            )

        preview = html.escape(
            preview
        )

        page_text = (
            f"Page {page}"
            if page is not None
            else "Page unavailable"
        )

        source_html += f"""
        <div class="source-item">

            <div class="source-file">
                ◇ {file_name}
            </div>

            <div class="source-page">
                {page_text}
            </div>

            <div class="source-preview">
                {preview}
            </div>

        </div>
        """

    source_html += """
    </div>
    """

    st.html(
        source_html
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="brand">

            <div class="brand-mark">
                ◆
            </div>

            <div class="brand-name">
                DocuMind AI
            </div>

        </div>

        <div class="brand-subtitle">
            Intelligent Document Assistant
        </div>
        """
    )

    is_ready = (
        st.session_state.vector_store
        is not None
    )

    status_text = (
        "SYSTEM READY"
        if is_ready
        else "WAITING FOR DOCUMENTS"
    )

    st.html(
        f"""
        <div class="system-status">

            <div class="status-left">

                <span class="status-dot"></span>

                {status_text}

            </div>

            <div class="status-version">
                v1.0
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="side-label">
            Knowledge Base
        </div>
        """
    )

    if st.session_state.documents:

        for file in st.session_state.documents:

            file_name = html.escape(
                file.name
            )

            size_mb = (
                len(
                    file.getvalue()
                )
                / (1024 * 1024)
            )

            st.html(
                f"""
                <div class="document-item">

                    <div class="document-icon">
                        ◇
                    </div>

                    <div class="document-info">

                        <div class="document-name">
                            {file_name}
                        </div>

                        <div class="document-meta">
                            {size_mb:.2f} MB · Indexed
                        </div>

                    </div>

                </div>
                """
            )

    else:

        st.html(
            """
            <div class="sidebar-empty">
                No documents yet.
                <br><br>
                Upload documents from the workspace.
            </div>
            """
        )

    st.html(
        """
        <div class="side-label">
            Workspace
        </div>
        """
    )

    document_count = len(
        st.session_state.documents
    )

    chunk_display = (
        f"{st.session_state.chunk_count:,}"
        if st.session_state.chunk_count
        else "—"
    )

    st.html(
        f"""
        <div class="workspace-grid">

            <div class="workspace-stat">

                <div class="workspace-value">
                    {document_count}
                </div>

                <div class="workspace-label">
                    Documents
                </div>

            </div>

            <div class="workspace-stat">

                <div class="workspace-value">
                    {chunk_display}
                </div>

                <div class="workspace-label">
                    Chunks
                </div>

            </div>

        </div>
        """
    )

    if st.session_state.vector_db_error:

        st.html(
            """
            <div class="sidebar-empty">
                FAISS is currently available
                in this session.
                <br><br>
                Disk persistence is temporarily
                unavailable.
            </div>
            """
        )


# ============================================================
# TOP BAR
# ============================================================

st.html(
    """
    <div class="topbar">

        <div class="topbar-left">

            Workspace /

            <span>
                Document Intelligence
            </span>

        </div>

        <div class="topbar-right">

            <div class="top-chip">
                ◆ Semantic Retrieval
            </div>

            <div class="top-chip">
                ◆ Hybrid Search
            </div>

            <div class="top-chip">
                ◆ BM25
            </div>

            <div class="top-chip">
                ◆ Reranking
            </div>

            <div class="top-chip">
                ◆ Page Citations
            </div>

        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-kicker">

            <span class="hero-kicker-dot"></span>

            Retrieval-Augmented Intelligence

        </div>

        <h1 class="hero-title">

            Understand your documents.<br>

            <span class="hero-gradient">
                Ask anything.
            </span>

        </h1>

        <div class="hero-description">

            A grounded AI workspace for exploring your documents,
            finding precise information, and tracing every answer
            back to its original source.

        </div>

        <div class="capabilities">

            <div class="capability">
                ◆ Semantic Retrieval
            </div>

            <div class="capability">
                ◆ Hybrid Search
            </div>

            <div class="capability">
                ◆ BM25
            </div>

            <div class="capability">
                ◆ Reranking
            </div>

            <div class="capability">
                ◆ Page-Level Citations
            </div>

        </div>

    </div>
    """
)


# ============================================================
# UPLOAD
# ============================================================

if not st.session_state.documents:

    st.html(
        """
        <div class="upload-workspace">

            <div class="upload-title">
                Build your knowledge base
            </div>

            <div class="upload-subtitle">
                Upload one or more PDF, DOCX, or TXT documents.
                Only files selected here will be indexed.
            </div>

            <div class="upload-formats">

                PDF
                <span>•</span>

                DOCX
                <span>•</span>

                TXT
                <span>•</span>

                MULTIPLE FILES SUPPORTED

            </div>

        </div>
        """
    )

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="main_upload",
    )

    if uploaded_files:

        signature = build_signature(
            uploaded_files
        )

        if (
            signature
            != st.session_state.index_signature
        ):

            try:

                with st.status(
                    "Building your knowledge base...",
                    expanded=True,
                ) as status:

                    st.write(
                        f"{len(uploaded_files)} "
                        "document(s) selected."
                    )

                    (
                        vector_store,
                        bm25,
                        documents,
                        chunk_count,
                    ) = build_knowledge_base(
                        uploaded_files
                    )

                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.bm25 = (
                        bm25
                    )

                    st.session_state.rag_documents = (
                        documents
                    )

                    st.session_state.documents = (
                        list(uploaded_files)
                    )

                    st.session_state.chunk_count = (
                        chunk_count
                    )

                    st.session_state.index_signature = (
                        signature
                    )

                    st.session_state.messages = []

                    status.update(
                        label=(
                            "Knowledge base ready · "
                            f"{chunk_count:,} chunks"
                        ),
                        state="complete",
                        expanded=False,
                    )

                st.rerun()

            except Exception as exc:

                st.error(
                    "Could not build the knowledge base."
                )

                st.exception(
                    exc
                )


# ============================================================
# KNOWLEDGE BASE READY
# ============================================================

else:

    st.html(
        f"""
        <div class="ready-card">

            <div class="ready-row">

                <div class="ready-icon">
                    ✓
                </div>

                <div>

                    <div class="ready-title">
                        Knowledge base ready
                    </div>

                    <div class="ready-subtitle">

                        {len(st.session_state.documents)}
                        document(s) indexed ·

                        {st.session_state.chunk_count:,}
                        chunks available for retrieval

                    </div>

                </div>

            </div>

        </div>
        """
    )

    additional_files = st.file_uploader(
        "Add additional documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="additional_upload",
    )

    if additional_files:

        combined_files = list(
            st.session_state.documents
        )

        existing_names = {
            file.name
            for file in combined_files
        }

        changed = False

        for file in additional_files:

            if file.name not in existing_names:

                combined_files.append(
                    file
                )

                existing_names.add(
                    file.name
                )

                changed = True

        if changed:

            combined_signature = (
                build_signature(
                    combined_files
                )
            )

            if (
                combined_signature
                != st.session_state.index_signature
            ):

                try:

                    with st.status(
                        "Updating knowledge base...",
                        expanded=True,
                    ) as status:

                        (
                            vector_store,
                            bm25,
                            documents,
                            chunk_count,
                        ) = build_knowledge_base(
                            combined_files
                        )

                        st.session_state.vector_store = (
                            vector_store
                        )

                        st.session_state.bm25 = (
                            bm25
                        )

                        st.session_state.rag_documents = (
                            documents
                        )

                        st.session_state.documents = (
                            combined_files
                        )

                        st.session_state.chunk_count = (
                            chunk_count
                        )

                        st.session_state.index_signature = (
                            combined_signature
                        )

                        st.session_state.messages = []

                        status.update(
                            label=(
                                "Knowledge base updated · "
                                f"{chunk_count:,} chunks"
                            ),
                            state="complete",
                            expanded=False,
                        )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "Could not update the knowledge base."
                    )

                    st.exception(
                        exc
                    )


# ============================================================
# EXPLORE
# ============================================================

st.html(
    """
    <div class="try-label">
        Explore your knowledge
    </div>
    """
)

col1, col2, col3 = st.columns(3)


with col1:

    st.html(
        """
        <div class="prompt-card">

            <div class="prompt-icon">
                ◆ SUMMARY
            </div>

            <div class="prompt-title">
                Summarize the key information
                across my documents
            </div>

        </div>
        """
    )


with col2:

    st.html(
        """
        <div class="prompt-card">

            <div class="prompt-icon">
                ◆ FIND
            </div>

            <div class="prompt-title">
                Find the most relevant information
                about a specific topic
            </div>

        </div>
        """
    )


with col3:

    st.html(
        """
        <div class="prompt-card">

            <div class="prompt-icon">
                ◆ COMPARE
            </div>

            <div class="prompt-title">
                Compare information across
                multiple documents
            </div>

        </div>
        """
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            render_sources(
                message["sources"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        if (
            st.session_state.vector_store is None
            or st.session_state.bm25 is None
            or not st.session_state.rag_documents
        ):

            answer = (
                "Please upload one or more documents "
                "before asking a question."
            )

            sources = []

            st.warning(
                answer
            )

        else:

            try:

                # ====================================================
                # 1. HYBRID RETRIEVAL
                # ====================================================

                with st.spinner(
                    "Searching your knowledge base..."
                ):

                    hybrid_results = hybrid_search(
                        vector_store=(
                            st.session_state.vector_store
                        ),
                        bm25=(
                            st.session_state.bm25
                        ),
                        documents=(
                            st.session_state.rag_documents
                        ),
                        query=question,
                        semantic_k=SEMANTIC_K,
                        bm25_k=BM25_K,
                        final_k=HYBRID_K,
                    )

                if not hybrid_results:

                    answer = (
                        "I couldn't find relevant information "
                        "in the uploaded documents."
                    )

                    sources = []

                    st.warning(
                        answer
                    )

                else:

                    # ====================================================
                    # 2. RERANK
                    # ====================================================

                    with st.spinner(
                        "Reranking relevant passages..."
                    ):

                        reranked_results = (
                            perform_reranking(
                                query=question,
                                results=hybrid_results,
                                top_k=RERANK_K,
                            )
                        )

                    # ====================================================
                    # 3. GENERATE ANSWER
                    # ====================================================

                    with st.spinner(
                        "Generating grounded answer..."
                    ):

                        response = (
                            generate_grounded_answer(
                                query=question,
                                results=reranked_results,
                            )
                        )

                    answer = response.get(
                        "answer",
                        "I couldn't generate an answer.",
                    )

                    sources = response.get(
                        "sources",
                        [],
                    )

                    st.markdown(
                        answer
                    )

                    # ====================================================
                    # 4. SOURCES
                    # ====================================================

                    if sources:

                        render_sources(
                            sources
                        )

            except Exception as exc:

                answer = (
                    "I couldn't process your question."
                )

                sources = []

                st.error(
                    answer
                )

                st.exception(
                    exc
                )

    # --------------------------------------------------------
    # SAVE CHAT HISTORY
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )