"""
Vector store for embeddings and metadata.
Provides functions to save and load embeddings from JSONL files.
"""

import json
import os
from typing import List, Dict, Any
from config import Config


def save_embeddings(data: List[Dict[str, Any]]) -> None:
    """
    Save embeddings and metadata to JSONL files.

    Args:
        data: List of dictionaries containing id, vector, text, and metadata
    """
    embeddings_path = Config.EMBEDDINGS_FILE
    metadata_path = Config.METADATA_FILE

    with open(embeddings_path, 'a', encoding='utf-8') as ef, \
         open(metadata_path, 'a', encoding='utf-8') as mf:

        for item in data:
            # Save embedding vector
            ef.write(json.dumps({
                'id': item['id'],
                'vector': item['vector']
            }) + '\n')

            # Save metadata
            mf.write(json.dumps({
                'id': item['id'],
                'text': item['text'],
                'metadata': item['metadata']
            }) + '\n')


def load_embeddings() -> List[Dict[str, Any]]:
    """
    Load all embeddings and metadata from JSONL files.

    Returns:
        List of dictionaries containing id, vector, text, and metadata
    """
    embeddings_path = Config.EMBEDDINGS_FILE
    metadata_path = Config.METADATA_FILE

    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        return []

    # Load embeddings into a dictionary by id
    embeddings = {}
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            embeddings[item['id']] = item['vector']

    # Load metadata and combine with embeddings
    results = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item_id = item['id']
            if item_id in embeddings:
                results.append({
                    'id': item_id,
                    'vector': embeddings[item_id],
                    'text': item['text'],
                    'metadata': item['metadata']
                })

    return results


def clear_index() -> None:
    """Delete all indexed data."""
    embeddings_path = Config.EMBEDDINGS_FILE
    metadata_path = Config.METADATA_FILE

    if os.path.exists(embeddings_path):
        os.remove(embeddings_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)


def index_exists() -> bool:
    """Check if an index exists."""
    return (os.path.exists(Config.EMBEDDINGS_FILE) and
            os.path.exists(Config.METADATA_FILE))
