# Latent-Space-Model Documentation Site

A Flask-based documentation site with RAG-powered "Ask Docs" chatbot for the Latent-Space-Model project.

## Features

- **Clean DigitalOcean-inspired design** with light/dark mode
- **Left sidebar navigation** with collapsible sections
- **Right sidebar TOC** for in-page navigation
- **RAG-powered chatbot** with Ollama embeddings and Groq generation
- **Comprehensive documentation** covering architecture, training, and usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Build the Vector Index

Before running the server, build the embedding index:

```bash
cd docs_site
python -m rag.index
```

This will:
- Read all markdown files from `data/corpus/`
- Chunk them using the ingestion pipeline
- Generate embeddings using Ollama with Qwen3-Embedding-4B
- Store results in `data/index/`

### 4. Run the Server

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) to view the documentation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Flask environment |
| `FLASK_DEBUG` | `1` | Enable debug mode |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M` | Embedding model |
| `RERANKER_MODEL` | `hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M` | Reranker model |
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM for generation |

## Architecture

### RAG Pipeline

1. **Ingestion** (`rag/ingest.py`): Load and chunk markdown files
2. **Indexing** (`rag/index.py`): Generate embeddings with Ollama
3. **Retrieval** (`rag/retrieve.py`): Cosine similarity search
4. **Reranking** (`rag/rerank.py`): Cross-encoder reranking
5. **Generation** (`rag/answer.py`): Groq API for answers

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main documentation page |
| `/api/health` | GET | Health check |
| `/api/ask` | POST | RAG question answering |
| `/api/chat` | POST | Chat endpoint (same as ask) |

## File Structure

```
docs_site/
├── app.py                  # Flask application
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── .env.example            # Environment template
├── rag/                    # RAG pipeline
│   ├── ingest.py          # Document chunking
│   ├── index.py           # Embedding generation
│   ├── retrieve.py        # Similarity search
│   ├── rerank.py          # Result reranking
│   ├── answer.py          # Groq answer generation
│   └── store.py           # Vector storage
├── data/
│   ├── corpus/            # Source documents
│   └── index/             # Generated embeddings
├── templates/
│   ├── layout.html        # Base template
│   └── index.html         # Documentation content
└── static/
    ├── css/site.css       # Styles
    └── js/chat.js         # Chatbot UI
```

## Customization

### Adding Documentation

Add markdown files to `data/corpus/` and rebuild the index:

```bash
python -m rag.index
```

### Modifying Appearance

Edit `static/css/site.css` to customize:
- Color scheme (CSS variables at top)
- Layout dimensions
- Typography
- Component styles

### Changing Embedding Model

Update `EMBEDDING_MODEL` in `.env` to any Ollama-compatible model.

## Troubleshooting

### Ollama Connection Error

Ensure Ollama is running locally:

```bash
ollama serve
```

And the required models are pulled:

```bash
ollama pull hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M
ollama pull hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M
```

### Groq API Error

Verify your `GROQ_API_KEY` is set correctly in `.env`.

### Index Not Found

Run the index build command:

```bash
python -m rag.index
```

## License

Same as the main Latent-Space-Model project.
