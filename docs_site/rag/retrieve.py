"""
Vector similarity search for the RAG pipeline.
Retrieves relevant chunks based on query embedding.
"""

import numpy as np
from typing import List, Tuple
import ollama
from config import Config
from rag.store import load_embeddings


def cosine_similarity(query_vec: List[float], doc_vecs: List[List[float]]) -> np.ndarray:
    """
    Compute cosine similarity between query and document vectors.

    Args:
        query_vec: Query embedding vector
        doc_vecs: List of document embedding vectors

    Returns:
        Array of similarity scores
    """
    query_arr = np.array(query_vec)
    doc_arr = np.array(doc_vecs)

    # Normalize vectors
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-8)
    doc_norms = doc_arr / (np.linalg.norm(doc_arr, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    similarities = np.dot(doc_norms, query_norm)

    return similarities


def embed_query(query: str) -> List[float]:
    """
    Embed a query string using the embedding model.

    Args:
        query: The query string

    Returns:
        Embedding vector
    """
    client = ollama.Client(host=Config.OLLAMA_HOST)

    try:
        response = client.embeddings(
            model=Config.EMBEDDING_MODEL,
            prompt=query
        )
        return response['embedding']
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []


def retrieve(query: str, top_k: int = None) -> List[dict]:
    """
    Retrieve the top-k most similar chunks for a query.

    Args:
        query: The query string
        top_k: Number of results to return

    Returns:
        List of chunk dictionaries with similarity scores
    """
    if top_k is None:
        top_k = Config.TOP_K_RETRIEVE

    # Load all embeddings
    all_embeddings = load_embeddings()

    if not all_embeddings:
        print("Warning: No embeddings found in index")
        return []

    # Embed the query
    query_vec = embed_query(query)

    if not query_vec:
        return []

    # Compute similarities
    doc_vecs = [item['vector'] for item in all_embeddings]
    similarities = cosine_similarity(query_vec, doc_vecs)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Build results
    results = []
    for idx in top_indices:
        item = all_embeddings[idx]
        results.append({
            'id': item['id'],
            'text': item['text'],
            'metadata': item['metadata'],
            'similarity': float(similarities[idx])
        })

    return results
