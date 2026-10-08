import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    """
    return re.findall(r"\b\w+\b", text.lower())


def get_documents_from_vector_store(vector_store) -> List[Document]:
    """
    Extract all documents/chunks already stored inside FAISS.
    """

    if not hasattr(vector_store, "docstore"):
        raise ValueError("FAISS vector store does not contain a docstore.")

    documents = list(vector_store.docstore._dict.values())

    print(f"Documents loaded from FAISS: {len(documents)}")

    return documents


def build_bm25(documents: List[Document]) -> BM25Okapi:
    """
    Build a BM25 index from existing document chunks.
    """

    print("\n" + "=" * 60)
    print("BUILDING BM25 INDEX")
    print("=" * 60)

    tokenized_documents = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    print(f"BM25 documents: {len(documents)}")
    print("BM25 index created.")

    return bm25


def bm25_search(
    bm25: BM25Okapi,
    documents: List[Document],
    query: str,
    k: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform BM25 keyword search.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):

        results.append({
            "document": documents[index],
            "bm25_score": float(scores[index]),
            "bm25_rank": rank
        })

    return results


def get_document_id(document: Document) -> str:
    """
    Create a stable ID for a chunk.
    """

    metadata = document.metadata

    if metadata.get("chunk_id") is not None:
        return str(metadata["chunk_id"])

    source = metadata.get("source", "")
    page = metadata.get("page_number", metadata.get("page", ""))
    content = document.page_content[:200]

    return f"{source}|{page}|{hash(content)}"


def semantic_search(
    vector_store,
    query: str,
    k: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform FAISS semantic search.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    output = []

    for rank, (document, score) in enumerate(results, start=1):

        output.append({
            "document": document,
            "semantic_score": float(score),
            "semantic_rank": rank
        })

    return output


def reciprocal_rank_fusion(
    semantic_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combine semantic and BM25 rankings using
    Reciprocal Rank Fusion (RRF).

    RRF formula:

        score = 1 / (k + rank)
    """

    fused = {}

    # -------------------------
    # Semantic results
    # -------------------------

    for result in semantic_results:

        document = result["document"]
        doc_id = get_document_id(document)

        if doc_id not in fused:
            fused[doc_id] = {
                "document": document,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "bm25_rank": None,
                "semantic_score": None,
                "bm25_score": None
            }

        fused[doc_id]["semantic_rank"] = result["semantic_rank"]
        fused[doc_id]["semantic_score"] = result["semantic_score"]

        fused[doc_id]["rrf_score"] += (
            1 / (rrf_k + result["semantic_rank"])
        )

    # -------------------------
    # BM25 results
    # -------------------------

    for result in bm25_results:

        document = result["document"]
        doc_id = get_document_id(document)

        if doc_id not in fused:
            fused[doc_id] = {
                "document": document,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "bm25_rank": None,
                "semantic_score": None,
                "bm25_score": None
            }

        fused[doc_id]["bm25_rank"] = result["bm25_rank"]
        fused[doc_id]["bm25_score"] = result["bm25_score"]

        fused[doc_id]["rrf_score"] += (
            1 / (rrf_k + result["bm25_rank"])
        )

    # -------------------------
    # Sort by RRF
    # -------------------------

    fused_results = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return fused_results


def hybrid_search(
    vector_store,
    bm25,
    documents,
    query: str,
    semantic_k: int = 15,
    bm25_k: int = 15,
    final_k: int = 10
):
    """
    Complete hybrid retrieval pipeline:

        Query
          ↓
        FAISS
          +
        BM25
          ↓
        RRF Fusion
          ↓
        Top K
    """

    print("\n" + "=" * 60)
    print("HYBRID SEARCH")
    print("=" * 60)

    print(f"Query: {query}")

    # Semantic retrieval
    semantic_results = semantic_search(
        vector_store,
        query,
        k=semantic_k
    )

    # BM25 retrieval
    bm25_results = bm25_search(
        bm25,
        documents,
        query,
        k=bm25_k
    )

    print(f"Semantic results: {len(semantic_results)}")
    print(f"BM25 results: {len(bm25_results)}")

    # Fusion
    fused_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results
    )

    final_results = fused_results[:final_k]

    print(f"Fused results: {len(final_results)}")

    return final_results


def print_hybrid_results(results):

    print("\n" + "=" * 80)
    print("HYBRID SEARCH RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document.metadata

        print(f"\nRESULT #{i}")
        print("-" * 60)

        print(f"RRF Score:       {result['rrf_score']:.6f}")
        print(f"Semantic Rank:   {result['semantic_rank']}")
        print(f"BM25 Rank:       {result['bm25_rank']}")
        print(f"Semantic Score:  {result['semantic_score']}")
        print(f"BM25 Score:      {result['bm25_score']}")

        print(f"Source:          {metadata.get('source')}")
        print(
            f"Page:            "
            f"{metadata.get('page_number', metadata.get('page'))}"
        )
        print(f"Chunk ID:         {metadata.get('chunk_id')}")

        print("\nContent:")
        print(document.page_content[:800])