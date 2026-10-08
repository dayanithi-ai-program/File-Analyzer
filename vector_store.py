import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_DB_PATH = "vector_db"


# ============================================================
# CHECK API KEY
# ============================================================

def check_api_key():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file."
        )

    return api_key


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def get_embeddings():

    check_api_key()

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return embeddings


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(
    chunks: List[Document],
    save_path: str = VECTOR_DB_PATH
):

    if not chunks:

        raise ValueError(
            "No document chunks were provided."
        )

    print()
    print("=" * 60)
    print("CREATING VECTOR STORE")
    print("=" * 60)

    print(
        f"Chunks to embed: {len(chunks)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print()

    # --------------------------------------------------------
    # Create embedding model
    # --------------------------------------------------------

    embeddings = get_embeddings()

    print(
        "Generating embeddings..."
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    vector_store = (
        FAISS.from_documents(
            chunks,
            embeddings
        )
    )

    print(
        "FAISS index created."
    )

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    Path(
        save_path
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    vector_store.save_local(
        save_path
    )

    print(
        f"Vector store saved to: "
        f"{save_path}"
    )

    print()
    print(
        "Vector store creation complete."
    )

    return vector_store


# ============================================================
# LOAD EXISTING VECTOR STORE
# ============================================================

def load_vector_store(
    save_path: str = VECTOR_DB_PATH
):

    embeddings = get_embeddings()

    if not Path(
        save_path
    ).exists():

        raise FileNotFoundError(
            f"Vector store not found: "
            f"{save_path}"
        )

    vector_store = (
        FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    )

    return vector_store


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    vector_store,
    query: str,
    k: int = 5
):

    if not query.strip():

        return []

    results = (
        vector_store.similarity_search_with_score(
            query,
            k=k
        )
    )

    return results


# ============================================================
# PRINT SEARCH RESULTS
# ============================================================

def print_search_results(
    results
):

    print()
    print("=" * 60)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 60)

    if not results:

        print(
            "No results found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        document, score = result

        print()
        print(
            f"RESULT #{rank}"
        )

        print(
            "-" * 60
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            "Source:",
            document.metadata.get(
                "source"
            )
        )

        print(
            "Page:",
            document.metadata.get(
                "page"
            )
        )

        print(
            "Chunk ID:",
            document.metadata.get(
                "chunk_id"
            )
        )

        print()

        print(
            document.page_content[:500]
        )

        print(
            "-" * 60
        )