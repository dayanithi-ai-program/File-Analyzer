import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI


RERANK_MODEL = "gpt-5-mini"


def rerank_documents(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved documents based on their relevance
    to the user's question.

    Input:
        Query + hybrid retrieval results

    Output:
        Top K most relevant documents
    """

    if not results:
        return []

    # Build candidates for the reranker
    candidates = []

    for i, result in enumerate(results):

        document = result["document"]
        metadata = document.metadata

        candidates.append({
            "candidate_id": i,
            "page": metadata.get(
                "page_number",
                metadata.get("page")
            ),
            "chunk_id": metadata.get("chunk_id"),
            "content": document.page_content
        })

    prompt = f"""
You are a document retrieval reranker.

Your job is to rank document chunks according to how directly
they answer the user's question.

IMPORTANT:
- Prefer chunks that directly answer the question.
- Do not rank a chunk highly merely because it contains similar
  keywords.
- Penalize table-of-contents pages.
- Penalize navigation/reference pages when they do not answer
  the question.
- Distinguish the requested concept from related concepts.
- For example, "Incident Management" is different from
  "Incident Communications Management" and "Major Incident".
- Return ONLY valid JSON.

USER QUESTION:
{query}

DOCUMENT CANDIDATES:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Return this exact JSON structure:

{{
    "ranking": [
        {{
            "candidate_id": 0,
            "relevance_score": 0,
            "reason": "short explanation"
        }}
    ]
}}

Rank ALL candidates from most relevant to least relevant.
Use relevance_score from 0 to 100.
"""

    llm = ChatOpenAI(
        model=RERANK_MODEL,
        temperature=0
    )

    response = llm.invoke(prompt)

    content = response.content

    # Handle models that return structured content blocks
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in content
        )

    content = content.strip()

    # Remove accidental markdown fences
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    ranking_data = json.loads(content)

    ranking = ranking_data.get("ranking", [])

    # Create lookup for original results
    result_lookup = {
        i: result
        for i, result in enumerate(results)
    }

    reranked_results = []

    for item in ranking:

        candidate_id = item.get("candidate_id")

        if candidate_id not in result_lookup:
            continue

        result = result_lookup[candidate_id].copy()

        result["rerank_score"] = float(
            item.get("relevance_score", 0)
        )

        result["rerank_reason"] = item.get(
            "reason",
            ""
        )

        reranked_results.append(result)

    # Sort by reranker score
    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


def print_reranked_results(results):

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document.metadata

        print(f"\nRESULT #{i}")
        print("-" * 60)

        print(
            f"Rerank Score: "
            f"{result.get('rerank_score')}"
        )

        print(
            f"RRF Score: "
            f"{result.get('rrf_score')}"
        )

        print(
            f"Semantic Rank: "
            f"{result.get('semantic_rank')}"
        )

        print(
            f"BM25 Rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Source: "
            f"{metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page_number', metadata.get('page'))}"
        )

        print(
            f"Chunk ID: "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Reason: "
            f"{result.get('rerank_reason')}"
        )

        print("\nContent:")
        print(document.page_content[:800])