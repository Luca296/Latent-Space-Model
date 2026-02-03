"""
Reranking using cross-encoder model via Ollama.
Refines initial retrieval results for better relevance.
"""

import re
from typing import List
import ollama
from config import Config


def rerank(query: str, candidates: List[dict], top_n: int = None) -> List[dict]:
    """
    Rerank candidates using a cross-encoder model.

    Args:
        query: The original query
        candidates: List of candidate chunks from initial retrieval
        top_n: Number of results to return after reranking

    Returns:
        Reranked list of candidates
    """
    if top_n is None:
        top_n = Config.TOP_N_RERANK

    if not candidates:
        return []

    # If we have fewer candidates than top_n, return all
    if len(candidates) <= top_n:
        return candidates

    client = ollama.Client(host=Config.OLLAMA_HOST)

    # Use Qwen3-Reranker for scoring
    scored_candidates = []

    for candidate in candidates:
        # Create prompt for reranking
        prompt = f"Query: {query}\n\nDocument: {candidate['text'][:500]}\n\nRelevance score (0-10):"

        try:
            # Get score from reranker model
            response = client.generate(
                model=Config.RERANKER_MODEL,
                prompt=prompt,
                options={'temperature': 0.0, 'num_predict': 10, 'num_ctx': Config.OLLAMA_NUM_CTX}
            )

            # Extract numeric score from response
            output = response.get('response', '').strip()
            # Try to find a number in the output
            number_match = re.search(r'(\d+(?:\.\d+)?)', output)
            if number_match:
                score = float(number_match.group(1))
                # Normalize to 0-1 range if needed
                if score > 1:
                    score = score / 10.0
            else:
                # Fallback: use original similarity
                score = candidate.get('similarity', 0)

        except Exception as e:
            print(f"Error reranking candidate {candidate['id']}: {e}")
            # Fallback to original similarity
            score = candidate.get('similarity', 0)

        scored_candidates.append({
            **candidate,
            'rerank_score': score
        })

    # Sort by rerank score
    scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    return scored_candidates[:top_n]
