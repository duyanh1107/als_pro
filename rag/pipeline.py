from __future__ import annotations

import json
import sys
from pathlib import Path

# Support both `python -m rag.pipeline` and direct `python pipeline.py`.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from rag.chunking import analyze_chunk_sizes, chunk_by_subsection_fallback, split_large_chunks
    from rag.document_processor import process_pdf_to_documents
    from rag.rag_service import RAGService
    from rag.toc_provider import get_toc_path_for_pdf
    from rag.vector_store import build_vector_store, load_vector_store, save_vector_store
else:
    from .chunking import analyze_chunk_sizes, chunk_by_subsection_fallback, split_large_chunks
    from .document_processor import process_pdf_to_documents
    from .rag_service import RAGService
    from .toc_provider import get_toc_path_for_pdf
    from .vector_store import build_vector_store, load_vector_store, save_vector_store


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
INDEXES_DIR = BASE_DIR / "data" / "indexes"
CHUNK_MAX_CHARS = 8000
CHUNK_SPLIT_STRATEGY = "paragraph_midpoint"
# Match embedding input to the full chunk size so chunk vectors represent the
# whole stored chunk instead of only a truncated prefix.
EMBEDDING_MAX_CHARS = CHUNK_MAX_CHARS
EMBEDDING_MODEL = "text-embedding-3-small"
DENSE_METRIC = "cosine"
ANSWER_MODEL = "gpt-4o-mini"


def build_rag_pipeline(
    pdf_path: str,
    toc_path: str | Path | None = None,
) -> tuple[RAGService, list[dict], list[dict]]:
    index, stored_chunks, _, manifest = load_or_build_vector_store(pdf_path, toc_path=toc_path)
    return RAGService(index, stored_chunks, model=ANSWER_MODEL), manifest["documents"], stored_chunks


def get_index_dir_for_pdf(pdf_path: str | Path) -> Path:
    return INDEXES_DIR / Path(pdf_path).stem


def build_manifest(
    pdf_path: Path,
    toc_path: Path,
    documents: list[dict],
    chunks: list[dict],
) -> dict:
    return {
        "pdf_path": str(pdf_path.resolve()),
        "toc_path": str(toc_path.resolve()),
        "pdf_mtime": pdf_path.stat().st_mtime,
        "toc_mtime": toc_path.stat().st_mtime,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_split_strategy": CHUNK_SPLIT_STRATEGY,
        "embedding_max_chars": EMBEDDING_MAX_CHARS,
        "embedding_model": EMBEDDING_MODEL,
        "dense_metric": DENSE_METRIC,
        "answer_model": ANSWER_MODEL,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        # Keep the source documents in the manifest so a cached load does not need to rebuild them.
        "documents": documents,
    }


def is_cached_index_valid(index_dir: Path, pdf_path: Path, toc_path: Path) -> bool:
    required_files = [
        index_dir / "index.faiss",
        index_dir / "embeddings.npy",
        index_dir / "chunks.json",
        index_dir / "manifest.json",
    ]
    if not all(path.exists() for path in required_files):
        return False

    try:
        with (index_dir / "manifest.json").open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return (
        manifest.get("pdf_path") == str(pdf_path.resolve())
        and manifest.get("toc_path") == str(toc_path.resolve())
        and manifest.get("pdf_mtime") == pdf_path.stat().st_mtime
        and manifest.get("toc_mtime") == toc_path.stat().st_mtime
        and manifest.get("chunk_max_chars") == CHUNK_MAX_CHARS
        and manifest.get("chunk_split_strategy") == CHUNK_SPLIT_STRATEGY
        and manifest.get("embedding_max_chars") == EMBEDDING_MAX_CHARS
        and manifest.get("embedding_model") == EMBEDDING_MODEL
        and manifest.get("dense_metric") == DENSE_METRIC
        and manifest.get("answer_model") == ANSWER_MODEL
    )


def load_or_build_vector_store(
    pdf_path: str,
    toc_path: str | Path | None = None,
) -> tuple[object, list[dict], object, dict]:
    resolved_pdf_path = Path(pdf_path).resolve()
    resolved_toc_path = get_toc_path_for_pdf(resolved_pdf_path, toc_path)
    index_dir = get_index_dir_for_pdf(resolved_pdf_path)

    if is_cached_index_valid(index_dir, resolved_pdf_path, resolved_toc_path):
        print(f"Loading cached index from: {index_dir}")
        return load_vector_store(index_dir)

    # Convert a raw PDF into page-level documents enriched with TOC metadata.
    print(f"Building new index for: {resolved_pdf_path.name}")
    documents, _, _, _ = process_pdf_to_documents(str(resolved_pdf_path), toc_path=resolved_toc_path)

    # Build semantic chunks first, then split oversized chunks before embedding.
    semantic_chunks = chunk_by_subsection_fallback(documents)
    semantic_chunks = split_large_chunks(semantic_chunks, max_chars=CHUNK_MAX_CHARS)
    index, stored_chunks, embeddings = build_vector_store(
        semantic_chunks,
        embedding_max_chars=EMBEDDING_MAX_CHARS,
    )
    manifest = build_manifest(resolved_pdf_path, resolved_toc_path, documents, stored_chunks)
    save_vector_store(index, stored_chunks, embeddings, index_dir, manifest)

    return index, stored_chunks, embeddings, manifest


def resolve_pdf_path(argv: list[str]) -> Path:
    # A CLI argument always wins over auto-discovery.
    if len(argv) > 1:
        candidate = Path(argv[1])
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    # Reuse any single PDF dropped into the documents folder.
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {DOCUMENTS_DIR}")

    if len(pdf_files) == 1:
        return pdf_files[0]

    names = ", ".join(path.name for path in pdf_files)
    raise FileNotFoundError(
        f"Multiple PDF files found in {DOCUMENTS_DIR}. Pass one explicitly: {names}"
    )


def main() -> int:
    try:
        path = resolve_pdf_path(sys.argv)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if not path.exists() or not path.is_file():
        print(f"PDF file not found: {path}")
        return 1

    print("=" * 80)
    print("RAG PIPELINE: SUBSECTION CHUNKING")
    print("=" * 80)
    print(f"Using PDF: {path}")
    print(f"Using TOC: {get_toc_path_for_pdf(path)}")
    print(f"Chunk split strategy: {CHUNK_SPLIT_STRATEGY}")
    print(f"Dense retrieval metric: {DENSE_METRIC}")

    # This prepares indexing once, then the chatbot loop only performs retrieval + answer generation.
    rag_service, documents, semantic_chunks = build_rag_pipeline(str(path))

    print(f"\nLoaded {len(documents)} documents")
    analyze_chunk_sizes(semantic_chunks)

    print("\n" + "=" * 80)
    print("CHUNKING STATS")
    print("=" * 80)
    print(f"Semantic chunks (subsection-level): {len(semantic_chunks)}")

    if len(semantic_chunks) > 5:
        sample = semantic_chunks[5]
        print("\n=== SAMPLE CHUNK ===")
        print(f"Section: {sample['section']}")
        print(f"Subsection: {sample['subsection']}")
        print(f"Size: {len(sample['content'])}")
        print(sample["content"][:200])

    print("\n" + "=" * 80)
    print("CHATBOT READY (type 'quit' to exit)")
    print("=" * 80)

    while True:
        query = input("\nYou: ")

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        results = rag_service.search(query, k=3)

        print("\nTop sources:")
        for result in results:
            print(f"- {result['section']} | {result['subsection']}")

        answer = rag_service.generate_answer(query, results)
        print("\nAnswer:")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
