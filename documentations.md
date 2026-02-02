# Documentation Site Plan

This document defines a standalone Flask documentation site for the Latent-Space-Model project. It will live in its own subfolder, run independently, and include an “Ask Docs” sidebar chatbot with a local RAG pipeline powered by Ollama (embeddings + reranking) and Groq for answer generation. No emojis will be used; emoticons are allowed.

## Goals

- Provide a single documentation page that explains the full project.
- Use a clean, modern layout with example images (as placeholders or real assets).
- Include an “Ask Docs” sidebar chatbot for project Q&A.
- Keep the docs site isolated from the main web app.
- Use a RAG pipeline with:
  - Embeddings: hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M via Ollama Python API
  - Reranking: hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M via Ollama Python API
  - Answer generation: Groq AI (chat completion)

## Repository Context to Document

Primary sources:
- README.md (overview, setup, usage)
- Middle-Model-Structure.md (deep dive architecture)
- src/models.py (core model classes)
- src/train.py (training stages and losses)
- src/data.py (dataset, caching, quantization)
- src/inference.py (inference flows)
- src/config.py (configuration and defaults)

## Folder Structure (New Subfolder)

Create a new folder at the repo root:

```
/docs_site
  app.py
  config.py
  rag/
    __init__.py
    ingest.py
    index.py
    retrieve.py
    rerank.py
    answer.py
    store.py
  data/
    corpus/
      README.md
      Middle-Model-Structure.md
      stack.md
      hotfix.md
    index/
      embeddings.jsonl
      metadata.jsonl
  templates/
    layout.html
    index.html
  static/
    css/
      site.css
    js/
      chat.js
    images/
      architecture.png
      pipeline.png
      training.png
  requirements.txt
  .env.example
```

Notes:
- `docs_site` is fully isolated from `src/web.py` and current templates.
- `data/corpus` contains copies or symlinked markdown sources.
- `data/index` is the local vector store.
- `static/images` includes example images referenced in the doc page.

## Documentation Page Structure

Single-page documentation site with sections:

1. Hero
   - Project name, short mission statement.
   - CTA buttons: “Read Overview”, “Ask Docs”.

2. Overview
   - High-level description of the latent-space architecture.
   - Key design principles and summary.

3. Architecture
   - Diagram image (architecture.png).
   - Explanation of encoder, middle model, prefix adapter, decoder.
   - Reference Middle-Model-Structure.md content.

4. Middle Model Deep Dive
   - TransformerBlock + MiddleTransformer explanation.
   - Tensor shapes and flow summary.

5. Training Pipeline
   - Stage 1–4 training regimen.
   - Loss functions and objectives.
   - Diagram image (training.png).

6. Data + Caching
   - Cache pipeline, quantization, dequantization.
   - How data flows into training and inference.

7. Inference
   - Summarization flow and constraints.
   - Checkpoints, stop latent, and decoding.

8. Configuration
   - Key config knobs from src/config.py.
   - Expected defaults.

9. FAQ
   - Pre-seeded FAQ items and links to “Ask Docs”.

10. Ask Docs Sidebar
   - Persistent right sidebar chatbot.
   - Uses RAG pipeline to answer questions.

## UI and Styling Rules

- No emojis anywhere; emoticons only (e.g., ":)", ";)", ":D").
- Clean, modern, large typography with clear sections.
- Light and dark mode supported.
- Sidebar chatbot fixed to the right edge.

## Flask App Behavior

- `app.py` exposes:
  - `GET /` -> render `index.html`.
  - `POST /api/ask` -> RAG pipeline with JSON response.
  - `GET /api/health` -> simple status.
- `config.py` loads env vars and defaults.

## RAG Pipeline Details

### 1) Ingestion

- Source files from `data/corpus`.
- Chunking:
  - 500–800 tokens per chunk.
  - 80–120 token overlap.
- Store metadata: file path, section title, line range.

### 2) Embedding (Ollama)

- Use Ollama Python API with:
  - model: `hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M`
- Store vectors in JSONL:
  - `embeddings.jsonl` entries with id, vector, text, metadata.

### 3) Retrieval

- Cosine similarity against stored vectors.
- Return top-k (k=12).

### 4) Reranking (Ollama)

- Use `hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M`.
- Rerank top-k to top-n (n=4–6).

### 5) Answer Generation (Groq)

- Build prompt with:
  - System: role as project documentation assistant.
  - User: question.
  - Context: reranked chunks with citations.
- Call Groq Chat Completion API.
- Return:
  - `answer`: formatted markdown.
  - `sources`: list of files/sections.

## Ask Docs Sidebar Behavior

- UI panel with:
  - Chat history.
  - Input box and submit button.
  - Optional “show sources” toggle.
- Request flow:
  - POST /api/ask with {question, history}.
  - Stream or return full response.

## Configuration and Environment

`docs_site/.env.example`:

```
FLASK_ENV=development
FLASK_DEBUG=1
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M
RERANKER_MODEL=hf.co/mradermacher/Qwen3-Reranker-4B-GGUF:Q4_K_M
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-70b-versatile
```

## Build Steps (Implementation Plan)

1. Create `docs_site` folder with the structure above.
2. Implement `app.py` with Flask routes.
3. Build templates:
   - `layout.html` for base layout and sidebar.
   - `index.html` for content sections.
4. Create `site.css` and `chat.js`.
5. Implement RAG modules:
   - `ingest.py` and `index.py` for chunking + embeddings.
   - `retrieve.py` for similarity search.
   - `rerank.py` for reranking.
   - `answer.py` for Groq responses.
6. Provide a script to rebuild the index.
7. Provide a startup guide in `docs_site/README.md`.

## Example Content and Images

Example image slots:
- `architecture.png`: overall system architecture.
- `pipeline.png`: training and inference pipeline.
- `training.png`: staged training diagram.

These can be placeholders first, then replaced.

## Security and Privacy

- Do not log user questions to disk by default.
- Redact API keys from logs.
- Provide a config toggle for logging.

## Testing Checklist

- Site loads and renders all sections.
- Ask Docs works with a basic question.
- RAG pipeline retrieves relevant chunks.
- Groq answers include citations.
- No emojis appear on the page.

## Deliverables

- New `docs_site` folder with all assets.
- `documentations.md` (this file) at repo root.
- Minimal instructions to run the docs server.

## Notes

- The docs site will not modify the existing Flask app in src/web.py.
- Emoticons can be used in text if desired, e.g., ":)", ":D".
