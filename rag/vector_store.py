from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embedding import embed_text


def _normalize_embedding_vectors(vectors: np.ndarray) -> np.ndarray:
    # Cosine search with FAISS is implemented by L2-normalizing vectors and using
    # inner product search. This keeps ranking aligned with cosine similarity.
    normalized = vectors.copy()
    faiss.normalize_L2(normalized)
    return normalized


def truncate_text_for_embedding(text: str, max_chars: int) -> str:
    # This remains as a safety guard if embedding_max_chars is set lower than the
    # stored chunk size in the future; with the current config, truncation should
    # normally not fire because embedding_max_chars matches CHUNK_MAX_CHARS.
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return truncated.rsplit(" ", 1)[0]


def build_vector_store(
    chunks: list[dict[str, Any]],
    embedding_max_chars: int,
) -> tuple[Any, list[dict[str, Any]], np.ndarray]:
    embeddings = []

    for chunk in chunks:
        # Each chunk is embedded independently so retrieval can return the most relevant sections.
        content = truncate_text_for_embedding(chunk["content"], max_chars=embedding_max_chars)
        if len(content) < len(chunk["content"]):
            name = chunk["subsection"] or chunk["section"] or chunk["chapter"]
            print(f"Truncated chunk '{name}'.")
        embeddings.append(embed_text(content))

    embeddings_array = np.array(embeddings, dtype="float32")
    normalized_embeddings = _normalize_embedding_vectors(embeddings_array)
    # Use cosine-style retrieval by normalizing vectors and searching with inner
    # product, which is equivalent to cosine similarity on unit-length vectors.
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)

    return index, chunks, normalized_embeddings


def search(
    query: str,
    index: Any,
    chunks: list[dict[str, Any]],
    k: int = 3,
) -> list[dict[str, Any]]:
    # Normalize the query embedding the same way as stored chunk vectors so
    # FAISS inner product ranking corresponds to cosine similarity.
    query_emb = _normalize_embedding_vectors(np.array([embed_text(query)], dtype="float32"))
    _, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]


def search_with_scores(
    query: str,
    index: Any,
    chunks: list[dict[str, Any]],
    k: int = 3,
) -> list[tuple[dict[str, Any], float]]:
    # With normalized vectors and IndexFlatIP, FAISS returns cosine-like inner
    # product scores directly, so larger scores are better dense matches.
    query_emb = _normalize_embedding_vectors(np.array([embed_text(query)], dtype="float32"))
    similarities, indices = index.search(query_emb, k)

    ranked: list[tuple[dict[str, Any], float]] = []
    for similarity, chunk_index in zip(similarities[0], indices[0], strict=False):
        if chunk_index < 0:
            continue
        ranked.append((chunks[int(chunk_index)], float(similarity)))
    return ranked


def save_vector_store(
    index: Any,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    index_dir: str | Path,
    manifest: dict[str, Any],
) -> None:
    resolved_dir = Path(index_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(resolved_dir / "index.faiss"))
    np.save(resolved_dir / "embeddings.npy", embeddings)

    with (resolved_dir / "chunks.json").open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    with (resolved_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def load_vector_store(
    index_dir: str | Path,
) -> tuple[Any, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    resolved_dir = Path(index_dir)
    index = faiss.read_index(str(resolved_dir / "index.faiss"))
    embeddings = np.load(resolved_dir / "embeddings.npy")

    with (resolved_dir / "chunks.json").open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with (resolved_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    return index, chunks, embeddings, manifest
