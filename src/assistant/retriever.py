"""Vector store retriever for the medical knowledge base (M3).

Indexes documents from data/kb/ using ChromaDB and retrieves
relevant context for user queries.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def _get_embeddings() -> HuggingFaceEmbeddings:
    # Suppress harmless checkpoint key mismatch log from transformers/BertModel
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)


def build_vectorstore(kb_dir: Path, persist_dir: Path) -> Chroma:
    """Load .txt files from kb_dir, chunk, embed, and persist to persist_dir.

    Args:
        kb_dir: Directory containing .txt knowledge base files.
        persist_dir: Directory where ChromaDB will persist the index.

    Returns:
        Loaded Chroma vectorstore.
    """
    loader = DirectoryLoader(str(kb_dir), glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=str(persist_dir),
    )
    return vectorstore


def get_retriever(persist_dir: Path, k: int = 3):
    """Load an existing ChromaDB vectorstore and return a top-k retriever.

    Args:
        persist_dir: Directory where the ChromaDB index is persisted.
        k: Number of documents to retrieve per query.

    Returns:
        VectorStoreRetriever configured for top-k search.
    """
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=_get_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
