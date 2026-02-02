"""
RAG pipeline modules.
"""

from rag.ingest import ingest_all, load_documents, chunk_document
from rag.index import build_index, embed_chunks
from rag.retrieve import retrieve
from rag.rerank import rerank
from rag.answer import generate_answer
from rag.store import load_embeddings, save_embeddings, clear_index, index_exists

__all__ = [
    'ingest_all',
    'load_documents',
    'chunk_document',
    'build_index',
    'embed_chunks',
    'retrieve',
    'rerank',
    'generate_answer',
    'load_embeddings',
    'save_embeddings',
    'clear_index',
    'index_exists',
]
