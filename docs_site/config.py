"""
Configuration for the Latent-Space-Model documentation site.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the Flask app and RAG pipeline."""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'

    # Ollama settings for embeddings
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    EMBEDDING_MODEL = os.environ.get(
        'EMBEDDING_MODEL',
        'hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M'
    )
    RERANKER_MODEL = os.environ.get(
        'RERANKER_MODEL',
        'hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M'
    )

    # Groq settings for generation
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')

    # RAG settings
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 120
    TOP_K_RETRIEVE = 12
    TOP_N_RERANK = 6

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CORPUS_DIR = os.path.join(BASE_DIR, 'data', 'corpus')
    INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
    EMBEDDINGS_FILE = os.path.join(INDEX_DIR, 'embeddings.jsonl')
    METADATA_FILE = os.path.join(INDEX_DIR, 'metadata.jsonl')


# Create directories if they don't exist
os.makedirs(Config.CORPUS_DIR, exist_ok=True)
os.makedirs(Config.INDEX_DIR, exist_ok=True)
