"""
Embedding and indexing for the RAG pipeline.
Uses Ollama to embed chunks and store them in the vector store.
"""

import time
from typing import List, Tuple
import ollama
from config import Config
from rag.ingest import Chunk, ingest_all
from rag.store import save_embeddings, clear_index


def embed_chunks(chunks: List[Chunk], batch_size: int = 8) -> List[Tuple[Chunk, List[float]]]:
    """
    Embed chunks using Ollama embedding model.

    Args:
        chunks: List of Chunk objects to embed
        batch_size: Number of chunks to embed per batch

    Returns:
        List of (chunk, embedding_vector) tuples
    """
    results = []
    client = ollama.Client(host=Config.OLLAMA_HOST)

    print(f"Embedding {len(chunks)} chunks using {Config.EMBEDDING_MODEL}...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        for chunk in batch:
            try:
                response = client.embeddings(
                    model=Config.EMBEDDING_MODEL,
                    prompt=chunk.text
                )

                if 'embedding' in response:
                    results.append((chunk, response['embedding']))
                else:
                    print(f"Warning: No embedding returned for chunk {chunk.chunk_id}")

            except Exception as e:
                print(f"Error embedding chunk {chunk.chunk_id}: {e}")

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Embedded {i + len(batch)}/{len(chunks)} chunks...")

        # Rate limiting to avoid overwhelming Ollama
        time.sleep(0.1)

    print(f"Successfully embedded {len(results)} chunks")
    return results


def build_index(corpus_dir: str = None) -> int:
    """
    Build the vector index from all corpus documents.

    Args:
        corpus_dir: Directory containing markdown files

    Returns:
        Number of chunks indexed
    """
    # Clear existing index
    clear_index()
    print("Cleared existing index")

    # Ingest documents
    chunks = ingest_all(corpus_dir)
    print(f"Created {len(chunks)} chunks from corpus")

    if not chunks:
        print("No chunks to index!")
        return 0

    # Embed chunks
    embedded = embed_chunks(chunks)

    # Prepare data for storage
    data = []
    for chunk, vector in embedded:
        data.append({
            'id': chunk.chunk_id,
            'vector': vector,
            'text': chunk.text,
            'metadata': {
                'source': chunk.source,
                'title': chunk.title,
                'section': chunk.section,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line
            }
        })

    # Save to store
    save_embeddings(data)
    print(f"Index built successfully with {len(data)} chunks")

    return len(data)


if __name__ == '__main__':
    count = build_index()
    print(f"\nIndex contains {count} chunks")
