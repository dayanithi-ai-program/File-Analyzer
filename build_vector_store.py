from rag_engine import load_document, create_chunks
from vector_store import create_vector_store


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "s.pdf"


# ============================================================
# LOAD PDF
# ============================================================

print()
print("=" * 60)
print("DOCUMIND AI")
print("BUILDING VECTOR DATABASE")
print("=" * 60)


with open(
    PDF_PATH,
    "rb"
) as f:

    file_bytes = f.read()


# ============================================================
# EXTRACT DOCUMENT
# ============================================================

documents = load_document(
    file_bytes,
    PDF_PATH
)


print()
print(
    f"Pages extracted: "
    f"{len(documents)}"
)


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = create_chunks(
    documents
)


print()
print(
    f"Chunks created: "
    f"{len(chunks)}"
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

vector_store = create_vector_store(
    chunks
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print(
    "FAISS vector database created."
)

print(
    "Location: vector_db/"
)

print()