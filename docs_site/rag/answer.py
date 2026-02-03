"""
Answer generation using Groq API.
Builds prompts and generates answers from retrieved context.
"""

from typing import List, Dict, Optional, Generator
from groq import Groq
from config import Config


SYSTEM_PROMPT = """You are a helpful documentation assistant for the Latent-Space-Model project.
Answer questions based on the provided documentation context.
Be concise, accurate, and technical when appropriate.

If the context doesn't contain the answer, say so honestly.
Always cite your sources using the citation markers [1], [2], etc."""


def build_prompt(question: str, context_chunks: List[dict]) -> str:
    """
    Build a prompt for the LLM with question and context.

    Args:
        question: User's question
        context_chunks: Retrieved context chunks with metadata

    Returns:
        Formatted prompt string
    """
    prompt_parts = ["Context:"]

    for i, chunk in enumerate(context_chunks, 1):
        source = chunk['metadata'].get('source', 'Unknown')
        section = chunk['metadata'].get('section', 'Unknown')
        text = chunk['text'][:800]  # Truncate very long chunks

        prompt_parts.append(f"\n[{i}] Source: {source}, Section: {section}")
        prompt_parts.append(text)

    prompt_parts.append(f"\n\nQuestion: {question}")
    prompt_parts.append("\nProvide a helpful answer citing sources like [1], [2], etc.:")

    return "\n".join(prompt_parts)


def generate_answer(
    question: str,
    chunks: List[dict],
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, any]:
    """
    Generate an answer using Groq API.

    Args:
        question: User's question
        chunks: Retrieved and reranked context chunks
        history: Optional conversation history

    Returns:
        Dictionary with answer and sources
    """
    if not Config.GROQ_API_KEY:
        return {
            'answer': 'Error: GROQ_API_KEY not configured. Please set it in your .env file.',
            'sources': []
        }

    if not chunks:
        return {
            'answer': 'I could not find any relevant documentation to answer your question.',
            'sources': []
        }

    client = Groq(api_key=Config.GROQ_API_KEY)

    # Build messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT}
    ]

    # Add conversation history if provided
    if history:
        for msg in history[-4:]:  # Only use last 4 exchanges
            messages.append(msg)

    # Add current question with context
    prompt = build_prompt(question, chunks)
    messages.append({'role': 'user', 'content': prompt})

    try:
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9
        )

        answer = response.choices[0].message.content

        # Extract citations from answer
        sources = extract_sources(answer, chunks)

        return {
            'answer': answer,
            'sources': sources
        }

    except Exception as e:
        return {
            'answer': f'Error generating answer: {str(e)}',
            'sources': []
        }


def generate_answer_stream(
    question: str,
    chunks: List[dict],
    history: Optional[List[Dict[str, str]]] = None
) -> Generator[Dict[str, any], None, None]:
    """
    Stream an answer using Groq API.

    Yields dictionaries with:
      - type: "delta" with partial content
      - type: "sources" with extracted sources
      - type: "done" when finished
    """
    if not Config.GROQ_API_KEY:
        yield {
            'type': 'delta',
            'content': 'Error: GROQ_API_KEY not configured. Please set it in your .env file.'
        }
        yield {
            'type': 'sources',
            'sources': []
        }
        yield {'type': 'done'}
        return

    if not chunks:
        yield {
            'type': 'delta',
            'content': 'I could not find any relevant documentation to answer your question.'
        }
        yield {
            'type': 'sources',
            'sources': []
        }
        yield {'type': 'done'}
        return

    client = Groq(api_key=Config.GROQ_API_KEY)

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT}
    ]

    if history:
        for msg in history[-4:]:
            messages.append(msg)

    prompt = build_prompt(question, chunks)
    messages.append({'role': 'user', 'content': prompt})

    answer_parts: List[str] = []

    try:
        stream = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ''
            if delta:
                answer_parts.append(delta)
                yield {
                    'type': 'delta',
                    'content': delta
                }

        full_answer = ''.join(answer_parts)
        sources = extract_sources(full_answer, chunks)
        yield {
            'type': 'sources',
            'sources': sources
        }
        yield {'type': 'done'}

    except Exception as e:
        yield {
            'type': 'delta',
            'content': f'Error generating answer: {str(e)}'
        }
        yield {
            'type': 'sources',
            'sources': []
        }
        yield {'type': 'done'}


def extract_sources(answer: str, chunks: List[dict]) -> List[dict]:
    """
    Extract which sources were cited in the answer.

    Args:
        answer: Generated answer text
        chunks: All chunks that were provided as context

    Returns:
        List of cited sources with metadata
    """
    import re

    # Find citation patterns like [1], [2], etc.
    citations = re.findall(r'\[(\d+)\]', answer)
    cited_indices = set(int(c) for c in citations)

    sources = []
    for i in cited_indices:
        if 1 <= i <= len(chunks):
            chunk = chunks[i - 1]
            sources.append({
                'id': i,
                'source': chunk['metadata'].get('source', 'Unknown'),
                'section': chunk['metadata'].get('section', 'Unknown'),
                'preview': chunk['text'][:200] + '...'
            })

    return sources
