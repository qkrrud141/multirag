"""Multi-vector index that maps searchable summaries back to original content."""

import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document


def create_retriever(
    *,
    collection_name: str = "multimodal_rag",
    persist_directory: str | Path | None = None,
):
    """Create a Chroma summary index and an in-memory original-content store."""
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    from langchain_core.stores import InMemoryStore
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(Path(persist_directory).resolve())
        if persist_directory is not None
        else None,
    )
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key="doc_id",
    )


def add_documents(
    retriever,
    summaries: Sequence[str],
    originals: Sequence[str],
    *,
    source_type: str,
) -> list[str]:
    """Index summaries and store their originals under matching generated IDs."""
    if len(summaries) != len(originals):
        raise ValueError(
            f"Summary/original count mismatch: {len(summaries)} != {len(originals)}."
        )
    if not source_type.strip():
        raise ValueError("source_type must not be empty.")
    if not summaries:
        return []

    ids = [str(uuid.uuid4()) for _ in originals]
    summary_docs = [
        Document(
            page_content=summary,
            metadata={retriever.id_key: item_id, "source_type": source_type},
        )
        for summary, item_id in zip(summaries, ids)
    ]
    retriever.vectorstore.add_documents(summary_docs, ids=ids)
    retriever.docstore.mset(list(zip(ids, originals)))
    return ids
