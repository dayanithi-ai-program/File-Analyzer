from typing import List, Dict, Any

from langchain_openai import ChatOpenAI


ANSWER_MODEL = "gpt-5-mini"


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Convert reranked documents into context for the LLM.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document.metadata

        source = metadata.get("source", "Unknown")
        page = metadata.get(
            "page_number",
            metadata.get("page")
        )

        context_parts.append(
            f"""
SOURCE {i}
File: {source}
Page: {page}
Chunk ID: {metadata.get("chunk_id")}

CONTENT:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    query: str,
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate an answer using ONLY the retrieved context.
    """

    if not results:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the uploaded documents."
            ),
            "sources": []
        }

    context = build_context(results)

    prompt = f"""
You are DocuMind AI, a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not supported by the documents, say:
   "I couldn't find enough information in the uploaded documents."
4. Give a concise but useful answer.
5. When making factual claims, cite the relevant source
   using this format:

   [Source: filename, Page X]

6. If multiple pages support the answer, cite each relevant page.
7. Do not mention retrieval, embeddings, FAISS, BM25, RRF,
   reranking, or internal system details.
8. Treat the uploaded documents as the only source of truth.

USER QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

Now answer the user's question.
"""

    llm = ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=0
    )

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in answer
        )

    # Build structured source information
    sources = []

    seen = set()

    for result in results:

        document = result["document"]
        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            metadata.get("page")
        )

        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": source,
            "page": page,
            "chunk_id": metadata.get("chunk_id"),
            "rerank_score": result.get(
                "rerank_score"
            )
        })

    return {
        "answer": answer,
        "sources": sources
    }