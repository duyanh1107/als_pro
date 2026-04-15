from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .vector_store import search as dense_search

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
RRF_K = 60
DEFAULT_ALPHA = 0.7


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(chunks)
        self.doc_lengths: list[int] = []
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_freqs: Counter[str] = Counter()

        for chunk in chunks:
            # Build sparse lexical statistics once so hybrid search does not need
            # to re-tokenize the full corpus on every query.
            tokens = _tokenize(chunk.get("content", ""))
            term_freqs = Counter(tokens)
            self.doc_lengths.append(len(tokens))
            self.doc_term_freqs.append(term_freqs)
            self.doc_freqs.update(term_freqs.keys())

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )

    def search(self, query: str, top_k: int) -> list[int]:
        return [doc_index for doc_index, _ in self.search_with_scores(query, top_k=top_k)]

    def search_with_scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_terms = _tokenize(query)
        if not query_terms or not self.doc_count:
            return []

        scores: list[tuple[float, int]] = []
        for doc_index, term_freqs in enumerate(self.doc_term_freqs):
            score = self._score_document(query_terms, term_freqs, self.doc_lengths[doc_index])
            if score > 0:
                scores.append((score, doc_index))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [(doc_index, score) for score, doc_index in scores[:top_k]]

    def _score_document(
        self,
        query_terms: list[str],
        term_freqs: Counter[str],
        doc_length: int,
    ) -> float:
        score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue

            df = self.doc_freqs.get(term, 0)
            # BM25 rewards terms that appear in fewer documents while normalizing
            # by document length so long chunks do not dominate lexical ranking.
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / (self.avg_doc_length or 1.0))
            )
            score += idf * ((tf * (self.k1 + 1)) / denominator)
        return score


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Index:
    return BM25Index(chunks)


def retrieve_relevant_chunks(
    query: str,
    index: Any,
    chunks: list[dict[str, Any]],
    bm25_index: BM25Index | None = None,
    k: int = 3,
    alpha: float = DEFAULT_ALPHA,
    debug: bool = False,
) -> list[dict[str, Any]]:
    # Hybrid retrieval combines dense semantic search with sparse lexical search,
    # then fuses their ranked results so exact keyword hits and semantic matches
    # can both contribute without hand-tuning score scales.
    dense_k = max(k * 4, 10)
    dense_results = dense_search(query, index, chunks, k=dense_k)

    sparse_results: list[dict[str, Any]] = []
    if bm25_index is not None:
        sparse_indices = bm25_index.search(query, top_k=dense_k)
        sparse_results = [chunks[index] for index in sparse_indices]

    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=k,
        alpha=alpha,
    )
    if debug:
        _print_hybrid_retrieval_debug(
            query,
            dense_results,
            sparse_results,
            fused_results,
            alpha=alpha,
        )
    return fused_results


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int,
    alpha: float = DEFAULT_ALPHA,
    rank_constant: int = RRF_K,
) -> list[dict[str, Any]]:
    scores: dict[tuple[Any, ...], float] = {}
    chunks_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    alpha = max(0.0, min(1.0, alpha))

    # Weight the dense and sparse ranked lists differently while staying purely
    # rank-based. This preserves the robustness of RRF without switching to
    # score normalization.
    for rank, chunk in enumerate(dense_results, start=1):
        key = _chunk_key(chunk)
        chunks_by_key[key] = chunk
        scores[key] = scores.get(key, 0.0) + alpha / (rank_constant + rank)

    for rank, chunk in enumerate(sparse_results, start=1):
        key = _chunk_key(chunk)
        chunks_by_key[key] = chunk
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) / (rank_constant + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [chunks_by_key[key] for key, _ in fused[:k]]


def _chunk_key(chunk: dict[str, Any]) -> tuple[Any, ...]:
    return (
        chunk.get("chunk_group_id"),
        chunk.get("chunk_part_index"),
        chunk.get("chapter"),
        chunk.get("section"),
        chunk.get("subsection"),
        chunk.get("start_page"),
    )


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _print_hybrid_retrieval_debug(
    query: str,
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    fused_results: list[dict[str, Any]],
    alpha: float,
) -> None:
    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL DEBUG")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Fusion: weighted_rrf | dense_alpha={alpha:.2f} | sparse_alpha={1.0 - alpha:.2f}")
    _print_ranked_chunks("Top dense hits", dense_results)
    _print_ranked_chunks("Top BM25 hits", sparse_results)
    _print_ranked_chunks("Final fused hits", fused_results)


def _print_ranked_chunks(title: str, ranked_chunks: list[dict[str, Any]]) -> None:
    print(f"\n{title}:")
    if not ranked_chunks:
        print("none")
        return

    for rank, chunk in enumerate(ranked_chunks, start=1):
        section = chunk.get("section") or "Unknown section"
        subsection = chunk.get("subsection") or "Unknown subsection"
        page_range = chunk.get("page_range") or "unknown pages"
        print(f"{rank}. {section} | {subsection} | pages {page_range}")
