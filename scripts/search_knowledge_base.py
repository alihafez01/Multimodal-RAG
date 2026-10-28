"""Search product text and genuine image descriptions with keyword, semantic, or hybrid retrieval."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from digital_product_rag.answering import answer_query  # noqa: E402
from digital_product_rag.retrieval.hybrid_search import (  # noqa: E402
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
    search_hybrid,
)
from digital_product_rag.retrieval.keyword_search import (  # noqa: E402
    KeywordSearchError,
    KeywordSearchIndex,
    has_query_terms,
)
from digital_product_rag.retrieval.vector_store import VectorStoreError, search as semantic_search  # noqa: E402


def _page_label(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return parsed


def _excerpt(text: str, limit: int = 360) -> str:
    text = text.translate({ord("●"): "-", ord("•"): "-", ord("▪"): "-"})
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _score_text(result: object) -> str:
    mode = getattr(result, "retrieval_mode", "semantic")
    if mode == "keyword":
        return f"BM25={result.bm25_score:.4f}"
    if mode == "hybrid":
        parts = [f"RRF={result.rrf_score:.6f}"]
        if result.semantic_score is not None:
            parts.append(f"cosine={result.semantic_score:.4f} (rank {result.semantic_rank})")
        if result.bm25_score is not None:
            parts.append(f"BM25={result.bm25_score:.4f} (rank {result.bm25_rank})")
        return " | ".join(parts)
    score = result.semantic_score if result.semantic_score is not None else result.similarity_score
    return f"cosine={score:.4f}"


def _display_results(title: str, results: Sequence[object], *, citations: bool) -> None:
    print(f"\n{title} ({len(results)} result{'s' if len(results) != 1 else ''})")
    if not results:
        print("  No results matched the query and filters.")
        return
    for position, result in enumerate(results, start=1):
        kind = "image_description" if result.chunk_type == "image_description" else "text"
        section = result.section_title or "(no section title)"
        reference = f" [C{position}]" if citations else ""
        print(
            f"\n[{position}]{reference} {_score_text(result)} | {kind} | "
            f"{result.product_title} ({result.product_id})"
        )
        print(
            f"  Source: {result.filename} | document={result.document_id} | "
            f"path={result.relative_path} | pages={_page_label(result.page_start, result.page_end)} | "
            f"section={section} | chunk={result.chunk_id}"
        )
        if result.chunk_type == "image_description":
            print(f"  Asset: {result.asset_id} | {result.asset_path}")
        print(f"  {_excerpt(result.text)}")


def _display_answer(response: object) -> None:
    if response.abstained:
        print("\nAnswer: I don't have enough relevant retrieved evidence to answer, so I am abstaining.")
        return
    if response.status == "answered":
        print(f"\nAnswer (local model {response.model}):\n{response.answer}")
        return
    if response.status == "evidence_only":
        print("\nEvidence-only mode: no LLM was used; the ranked passages above are the cited evidence.")
        return
    print(f"\nLLM answer unavailable ({response.status}): {response.error}")
    print("Showing retrieved evidence only; no unvalidated answer was displayed.")


def _display_citation_sources(response: object) -> None:
    if not response.citations:
        return
    print("\nCitation sources")
    for item in response.citations:
        pages = ", ".join(str(page) for page in item.source_pages) or _page_label(item.page_start, item.page_end)
        print(
            f"  [{item.reference}] {item.product_title} | {item.filename} | pages {pages} | "
            f"chunk={item.chunk_id} | document={item.document_id}"
        )
        if item.chunk_type == "image_description":
            print(f"      asset_id={item.asset_id} | asset_path={item.asset_path}")


def main(argv: list[str] | None = None) -> int:
    # Replace unsupported source glyphs rather than failing on legacy PowerShell code pages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--mode", choices=("semantic", "keyword", "hybrid"), default="hybrid")
    parser.add_argument("--product-id")
    parser.add_argument("--document-id")
    parser.add_argument("--chunk-type", choices=("text", "image_description"))
    parser.add_argument("--candidate-limit", type=_positive_int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--rrf-k", type=_positive_int, default=DEFAULT_RRF_K)
    parser.add_argument("--compare", action="store_true", help="Display semantic, keyword, and hybrid sections.")
    parser.add_argument("--answer-mode", choices=("evidence", "llm"), default="evidence")
    parser.add_argument("--llm-model", help="Exact local Ollama model name (or set OLLAMA_CHAT_MODEL).")
    parser.add_argument("--ollama-host", default=None, help="Local loopback Ollama URL; defaults to OLLAMA_HOST or 127.0.0.1:11434.")
    parser.add_argument("--llm-timeout", type=_positive_float, default=120.0)
    args = parser.parse_args(argv)

    filters = {
        "product_id": args.product_id,
        "document_id": args.document_id,
        "chunk_type": args.chunk_type,
    }
    try:
        if args.compare:
            keyword_index = KeywordSearchIndex(REPOSITORY_ROOT / "data" / "processed" / "chunks.jsonl")
            keyword_index.validate_filters(**filters)
            filtered = any(value is not None for value in filters.values())
            candidate_count = max(args.top_k, len(keyword_index.chunks) if filtered else args.candidate_limit)
            semantic_results = semantic_search(
                REPOSITORY_ROOT, args.query, top_k=candidate_count, **filters
            )
            keyword_results = keyword_index.search(args.query, top_k=candidate_count, **filters)
            hybrid_results = (
                reciprocal_rank_fusion(
                    semantic_results, keyword_results, top_k=args.top_k, rrf_k=args.rrf_k
                )
                if has_query_terms(args.query) else []
            )
            _display_results("Semantic search", semantic_results[:args.top_k], citations=False)
            _display_results("Keyword search", keyword_results[:args.top_k], citations=False)
            _display_results("Hybrid search (reciprocal rank fusion)", hybrid_results, citations=True)
            evidence_results = hybrid_results
            used_mode = "hybrid"
        elif args.mode == "semantic":
            results = semantic_search(REPOSITORY_ROOT, args.query, top_k=args.top_k, **filters)
            _display_results("Semantic search", results, citations=True)
            evidence_results = results
            used_mode = "semantic"
        elif args.mode == "keyword":
            keyword_index = KeywordSearchIndex(REPOSITORY_ROOT / "data" / "processed" / "chunks.jsonl")
            results = keyword_index.search(args.query, top_k=args.top_k, **filters)
            _display_results("Keyword search", results, citations=True)
            evidence_results = results
            used_mode = "keyword"
        else:
            results = search_hybrid(
                REPOSITORY_ROOT,
                args.query,
                top_k=args.top_k,
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
                **filters,
            )
            _display_results("Hybrid search (reciprocal rank fusion)", results, citations=True)
            evidence_results = results
            used_mode = "hybrid"
    except (KeywordSearchError, VectorStoreError, ValueError, OSError, RuntimeError) as exc:
        print(f"Search failed in the requested mode: {exc}", file=sys.stderr)
        return 2

    response = answer_query(
        args.query,
        evidence_results,
        mode=args.answer_mode,
        model=args.llm_model,
        ollama_host=args.ollama_host,
        timeout=args.llm_timeout,
    )
    print(f"\nAnswer mode: {args.answer_mode}; evidence retrieval mode: {used_mode}")
    _display_answer(response)
    _display_citation_sources(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
