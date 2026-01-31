"""
Generate an interactive UMAP map from cached embeddings.

Parses all *.jsonl files in the cache folder and extracts embeddings from
fields named "embedding", "source_embedding", and "target_embedding".
If int8-quantized fields are present (e.g., "embedding_q" + "embedding_scale"),
it will dequantize to float16 before projection.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Any

import numpy as np
import umap
import plotly.express as px


def _dequantize_int8(emb_q: list, scale: float) -> np.ndarray:
    if scale is None or scale == 0:
        scale = 1.0
    emb_q_arr = np.array(emb_q, dtype=np.int8)
    return (emb_q_arr.astype(np.float16) * np.float16(scale)).astype(np.float16)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_embedding(item: dict, key: str) -> np.ndarray | None:
    if f"{key}_q" in item and f"{key}_scale" in item:
        return _dequantize_int8(item[f"{key}_q"], item[f"{key}_scale"])
    if key in item:
        return np.array(item[key], dtype=np.float16)
    return None


def load_embeddings(cache_dir: Path, max_rows: int | None = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    embeddings: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    jsonl_files = sorted(cache_dir.glob("*.jsonl"))
    for jsonl_path in jsonl_files:
        for idx, item in enumerate(_iter_jsonl(jsonl_path)):
            for key in ["embedding", "source_embedding", "target_embedding"]:
                emb = _extract_embedding(item, key)
                if emb is None:
                    continue
                embeddings.append(emb)
                meta.append({
                    "file": jsonl_path.name,
                    "index": idx,
                    "key": key,
                    "text": item.get("text") or item.get("source") or item.get("target") or ""
                })
                if max_rows is not None and len(embeddings) >= max_rows:
                    break
            if max_rows is not None and len(embeddings) >= max_rows:
                break
        if max_rows is not None and len(embeddings) >= max_rows:
            break

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    # Filter any mismatched dimensions
    target_dim = embeddings[0].shape[0]
    filtered_embeddings = []
    filtered_meta = []
    for emb, m in zip(embeddings, meta):
        if emb.shape[0] == target_dim:
            filtered_embeddings.append(emb)
            filtered_meta.append(m)

    emb_matrix = np.vstack(filtered_embeddings).astype(np.float32)
    return emb_matrix, filtered_meta


def build_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float, metric: str) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    return reducer.fit_transform(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UMAP HTML from cached embeddings")
    parser.add_argument("--cache-dir", type=str, default="cache/preprocessed", help="Path to cache folder with jsonl files")
    parser.add_argument("--output", type=str, default="cache/umap_embeddings.html", help="Output HTML file")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--metric", type=str, default="euclidean", help="UMAP metric")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on number of embeddings")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"Cache dir not found: {cache_dir}")

    embeddings, meta = load_embeddings(cache_dir, max_rows=args.max_rows)
    if embeddings.size == 0:
        raise SystemExit("No embeddings found in cache jsonl files.")

    emb2d = build_umap(embeddings, args.n_neighbors, args.min_dist, args.metric)

    plot_data = {
        "x": emb2d[:, 0],
        "y": emb2d[:, 1],
        "file": [m["file"] for m in meta],
        "index": [m["index"] for m in meta],
        "key": [m["key"] for m in meta],
        "text": [m["text"] for m in meta],
    }

    fig = px.scatter(
        plot_data,
        x="x",
        y="y",
        hover_data=["file", "index", "key", "text"],
        title="UMAP Projection of Cached Embeddings",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved interactive UMAP to {output_path}")


if __name__ == "__main__":
    main()
