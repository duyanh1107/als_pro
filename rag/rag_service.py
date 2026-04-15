from __future__ import annotations

from typing import Any

from llm.answer_prompt_builder import build_answer_generation_prompt
from llm.hyde_generator import generate_hypothetical_document
from llm.query_decomposer import decompose_query
from llm.query_expander import expand_query
from llm.reranker import DEFAULT_CROSS_ENCODER_MODEL, rerank_chunks
from llm.query_transform_selector import select_query_transform
from llm.step_back_prompting import step_back_query
from .embedding import get_openai_client
from .grounding import build_grounded_context
from .retrieval import build_bm25_index, retrieve_relevant_chunks

DEFAULT_HYBRID_CANDIDATE_K = 20
DEFAULT_RERANK_TOP_K = 3


class RAGService:
    def __init__(
        self,
        index: Any,
        chunks: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        reranker_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        answer_temperature: float = 0,
        fusion_alpha: float = 0.7,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ):
        # Keep the built FAISS index and source chunks together so retrieval results
        # can be fed straight into answer generation.
        self.index = index
        self.chunks = chunks
        self.model = model
        # Keep the cross-encoder model separate from the chat model because the
        # reranker loads a Hugging Face model, not an OpenAI chat completion model.
        self.reranker_model = reranker_model
        self.answer_temperature = answer_temperature
        self.fusion_alpha = fusion_alpha
        self.candidate_k = candidate_k
        self.rerank_top_k = rerank_top_k
        self.client = get_openai_client()
        # Precompute the sparse lexical index once so hybrid search can fuse BM25
        # and dense vector retrieval without rebuilding corpus statistics per query.
        self.bm25_index = build_bm25_index(chunks)

    def search(self, query: str, k: int | None = None, debug: bool = False) -> list[dict[str, Any]]:
        return retrieve_relevant_chunks(
            query,
            self.index,
            self.chunks,
            bm25_index=self.bm25_index,
            k=k or self.candidate_k,
            alpha=self.fusion_alpha,
            debug=debug,
        )

    def select_transform_and_search(
        self,
        query: str,
        module=None,
        k_per_query: int = DEFAULT_HYBRID_CANDIDATE_K,
        debug: bool = False,
    ) -> dict[str, Any]:
        # Route the learner question through a single transform selector so the
        # caller does not need to choose manually among the available strategies.
        selection = select_query_transform(
            query,
            module_title=getattr(module, "title", None),
            chapter_title=getattr(module, "chapter_title", None),
            model=self.model,
            debug=debug,
        )
        transform = selection["transform"]

        if transform == "expand":
            result = self.expand_and_search(query, k_per_query=k_per_query, module=module, debug=debug)
        elif transform == "decompose":
            result = self.decompose_and_search(query, k_per_subquery=k_per_query, module=module, debug=debug)
        elif transform == "step_back":
            result = self.step_back_and_search(query, k_per_query=k_per_query, module=module, debug=debug)
        elif transform == "hyde":
            result = self.hyde_and_search(query, k_per_query=k_per_query, module=module, debug=debug)
        else:
            # Even when no transform is selected, keep the cleaned query from the
            # selector so downstream logs and retrieval use typo-corrected wording.
            normalized_query = str(selection.get("normalized") or query).strip()
            result = {
                "query_info": {
                    "original": query,
                    "normalized": normalized_query,
                    "transform": "none",
                },
                **self._build_reranked_result(normalized_query, self.search(normalized_query, k=k_per_query, debug=debug), debug=debug),
            }

        result["selection"] = selection
        return result

    def expand_and_search(
        self,
        query: str,
        k_per_query: int = DEFAULT_HYBRID_CANDIDATE_K,
        module=None,
        debug: bool = False,
    ) -> dict[str, Any]:
        # Expand the learner query before semantic search so retrieval is more
        # tolerant of typos and alternate phrasings without changing the intent.
        expanded = expand_query(
            query,
            module_title=getattr(module, "title", None),
            chapter_title=getattr(module, "chapter_title", None),
            model=self.model,
            debug=debug,
        )
        return {
            "query_info": expanded,
            # Query expansion now produces one enriched retrieval query instead of
            # many alternate queries, so the expanded keywords stay in one search.
            **self._build_reranked_result(
                expanded["expanded_query"],
                self.search(expanded["expanded_query"], k=k_per_query, debug=debug),
                debug=debug,
            ),
        }

    def decompose_and_search(
        self,
        query: str,
        k_per_subquery: int = DEFAULT_HYBRID_CANDIDATE_K,
        module=None,
        debug: bool = False,
    ) -> dict[str, Any]:
        # Decompose multi-part questions into smaller retrieval targets before
        # merging the results back into one grounded context set.
        decomposition = decompose_query(
            query,
            module_title=getattr(module, "title", None),
            chapter_title=getattr(module, "chapter_title", None),
            model=self.model,
            debug=debug,
        )
        combined = self._merge_search_results(
            decomposition["subqueries"],
            k_per_query=k_per_subquery,
            debug=debug,
        )
        return {
            "query_info": decomposition,
            **self._build_reranked_result(decomposition["normalized"], combined, debug=debug),
        }

    def step_back_and_search(
        self,
        query: str,
        k_per_query: int = DEFAULT_HYBRID_CANDIDATE_K,
        module=None,
        debug: bool = False,
    ) -> dict[str, Any]:
        # Use both the original query and a higher-level abstraction to retrieve
        # local details plus broader conceptual context.
        stepped = step_back_query(
            query,
            module_title=getattr(module, "title", None),
            chapter_title=getattr(module, "chapter_title", None),
            model=self.model,
        )
        queries = [stepped["normalized"], stepped["step_back_query"]]
        combined = self._merge_search_results(queries, k_per_query=k_per_query, debug=debug)
        return {
            "query_info": stepped,
            **self._build_reranked_result(stepped["normalized"], combined, debug=debug),
        }

    def hyde_and_search(
        self,
        query: str,
        k_per_query: int = DEFAULT_HYBRID_CANDIDATE_K,
        module=None,
        debug: bool = False,
    ) -> dict[str, Any]:
        # HyDE retrieves against a hypothetical answer passage rather than only
        # the raw question wording, which can improve matches for implicit queries.
        hyde = generate_hypothetical_document(
            query,
            module_title=getattr(module, "title", None),
            chapter_title=getattr(module, "chapter_title", None),
            model=self.model,
        )
        queries = [hyde["normalized"], hyde["hypothetical_document"]]
        combined = self._merge_search_results(queries, k_per_query=k_per_query, debug=debug)
        return {
            "query_info": hyde,
            **self._build_reranked_result(hyde["normalized"], combined, debug=debug),
        }

    def retrieve_for_module(self, module, k: int = 5) -> list[dict[str, Any]]:
        # Module retrieval starts with metadata alignment because section/subsection
        # matches are usually more reliable than a fresh semantic search.
        normalized_module_title = module.title.strip().lower()
        normalized_chapter_title = (module.chapter_title or "").strip().lower()

        exact_matches = []
        chapter_matches = []
        for chunk in self.chunks:
            chunk_section = (chunk.get("section") or "").strip().lower()
            chunk_subsection = (chunk.get("subsection") or "").strip().lower()
            chunk_chapter = (chunk.get("chapter") or "").strip().lower()

            # Prefer exact metadata matches first so module retrieval does not depend on another embedding call.
            if chunk_section == normalized_module_title or chunk_subsection == normalized_module_title:
                if not normalized_chapter_title or chunk_chapter == normalized_chapter_title:
                    exact_matches.append(chunk)
                    continue

            # If we do not have enough exact hits, look for chunk names inside the same chapter.
            if normalized_chapter_title and chunk_chapter == normalized_chapter_title:
                if normalized_module_title in chunk_section or normalized_module_title in chunk_subsection:
                    chapter_matches.append(chunk)

        if exact_matches:
            combined = []
            seen = set()
            for chunk in exact_matches + chapter_matches:
                key = (
                    chunk.get("chunk_group_id"),
                    chunk.get("chunk_part_index"),
                    chunk.get("chapter"),
                    chunk.get("section"),
                    chunk.get("subsection"),
                    chunk.get("start_page"),
                )
                if key in seen:
                    continue
                seen.add(key)
                combined.append(chunk)
            # Return all exact subsection matches so large split sections are not cut off.
            # This is important because one logical subsection may have been split into
            # many physical chunks during preprocessing.
            return sorted(
                combined,
                key=lambda chunk: (
                    chunk.get("start_page", 0),
                    chunk.get("chunk_part_index", 0),
                ),
            )

        # Semantic fallback is only used when metadata matching cannot find the module content.
        semantic_query = " | ".join(
            part for part in [module.chapter_title, module.title, module.primary_skill] if part
        )
        semantic_matches = self.search(semantic_query, k=k)

        combined = []
        seen = set()
        for chunk in exact_matches + semantic_matches:
            # Use split-part metadata in the dedupe key so paragraph-split chunks
            # from the same subsection are preserved as distinct retrieval units.
            key = (
                chunk.get("chunk_group_id"),
                chunk.get("chunk_part_index"),
                chunk.get("chapter"),
                chunk.get("section"),
                chunk.get("subsection"),
                chunk.get("start_page"),
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(chunk)

        combined = sorted(
            combined,
            key=lambda chunk: (
                chunk.get("start_page", 0),
                chunk.get("chunk_part_index", 0),
            ),
        )
        return combined[:k]

    def generate_answer(self, query: str, chunks: list[dict[str, Any]]) -> str:
        grounded = build_grounded_context(query, chunks)
        prompt = build_answer_generation_prompt(
            query,
            grounded_context=grounded["context_text"],
            grounded_sources=grounded["sources"],
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.answer_temperature,
        )
        return response.choices[0].message.content or ""

    def _merge_search_results(
        self,
        queries: list[str],
        k_per_query: int,
        debug: bool = False,
    ) -> list[dict[str, Any]]:
        combined = []
        seen = set()
        for candidate in queries:
            if not candidate:
                continue
            for chunk in self.search(candidate, k=k_per_query, debug=debug):
                key = (
                    chunk.get("chunk_group_id"),
                    chunk.get("chunk_part_index"),
                    chunk.get("chapter"),
                    chunk.get("section"),
                    chunk.get("subsection"),
                    chunk.get("start_page"),
                )
                if key in seen:
                    continue
                seen.add(key)
                combined.append(chunk)

        return sorted(
            combined,
            key=lambda chunk: (
                chunk.get("start_page", 0),
                chunk.get("chunk_part_index", 0),
            ),
        )

    def _build_reranked_result(
        self,
        query: str,
        candidate_chunks: list[dict[str, Any]],
        debug: bool = False,
    ) -> dict[str, Any]:
        reranked = rerank_chunks(
            query,
            candidate_chunks,
            top_k=self.rerank_top_k,
            model=self.reranker_model,
            debug=debug,
        )
        grounded = build_grounded_context(query, reranked["chunks"])
        return {
            "candidate_chunks": candidate_chunks,
            "chunks": reranked["chunks"],
            "rerank_info": {
                "query": reranked["query"],
                "selected_indices": reranked["selected_indices"],
            },
            "grounded_sources": grounded["sources"],
            "grounded_context": grounded["context_text"],
        }
