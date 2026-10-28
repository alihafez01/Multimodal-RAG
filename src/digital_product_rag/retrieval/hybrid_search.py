"""Semantic, keyword, and rank-fused hybrid retrieval."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from ..schemas import SearchResult
from .keyword_search import KeywordSearchIndex, has_query_terms
from .vector_store import search as semantic_search


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_LIMIT = 50


class HybridSearchError(RuntimeError):
    """Raised when hybrid candidate rankings cannot be safely combined."""


def reciprocal_rank_fusion(
    semantic_results: Sequence[SearchResult],
    keyword_results: Sequence[SearchResult],
    *,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Combine one-based ranks, preserving single-list results and raw scores."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")

    def indexed(results: Sequence[SearchResult]) -> dict[str, tuple[int, SearchResult]]:
        output: dict[str, tuple[int, SearchResult]] = {}
        for rank, result in enumerate(results, start=1):
            if result.chunk_id not in output:
                output[result.chunk_id] = (rank, result)
        return output

    semantic = indexed(semantic_results)
    keyword = indexed(keyword_results)
    all_ids = set(semantic) | set(keyword)
    merged: list[SearchResult] = []
    for chunk_id in all_ids:
        sem_entry, kw_entry = semantic.get(chunk_id), keyword.get(chunk_id)
        sem_rank = sem_entry[0] if sem_entry else None
        kw_rank = kw_entry[0] if kw_entry else None
        sem_result = sem_entry[1] if sem_entry else None
        kw_result = kw_entry[1] if kw_entry else None
        base = sem_result or kw_result
        assert base is not None
        if sem_result is not None and kw_result is not None:
            for field in (
                "document_id", "product_id", "product_title", "product_category", "product_version",
                "chunk_type", "section_title", "text", "filename", "relative_path", "page_start",
                "page_end", "source_pages", "asset_id", "asset_path", "metadata_status",
            ):
                if getattr(sem_result, field) != getattr(kw_result, field):
                    raise HybridSearchError(
                        f"Retrievers returned conflicting {field} metadata for chunk {chunk_id!r}."
                    )
        semantic_score = (
            sem_result.semantic_score if sem_result and sem_result.semantic_score is not None
            else sem_result.similarity_score if sem_result else None
        )
        bm25_score = kw_result.bm25_score if kw_result else None
        rrf_score = (1.0 / (rrf_k + sem_rank) if sem_rank is not None else 0.0) + (
            1.0 / (rrf_k + kw_rank) if kw_rank is not None else 0.0
        )
        merged.append(replace(
            base,
            similarity_score=semantic_score,
            semantic_score=semantic_score,
            bm25_score=bm25_score,
            semantic_rank=sem_rank,
            bm25_rank=kw_rank,
            rrf_score=rrf_score,
            retrieval_mode="hybrid",
        ))
    merged.sort(key=lambda result: (-float(result.rrf_score or 0.0), result.chunk_id))
    return merged[:top_k]


def search_hybrid(
    repository_root: Path,
    query: str,
    *,
    top_k: int = 5,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
    product_id: str | None = None,
    document_id: str | None = None,
    chunk_type: str | None = None,
    model_name: str | None = None,
    encoder: object | None = None,
    index_bundle: tuple[object, dict[int, dict[str, object]], dict[str, object]] | None = None,
    keyword_index: KeywordSearchIndex | None = None,
    semantic_retriever: Callable[..., list[SearchResult]] = semantic_search,
) -> list[SearchResult]:
    """Retrieve from both existing indexes and combine their rank positions with RRF."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit < 1:
        raise ValueError("candidate_limit must be a positive integer.")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")

    index = keyword_index or KeywordSearchIndex(
        repository_root / "data" / "processed" / "chunks.jsonl"
    )
    filters = {
        "product_id": product_id,
        "document_id": document_id,
        "chunk_type": chunk_type,
    }
    index.validate_filters(**filters)
    if not has_query_terms(query):
        return []

    filtered = any(value is not None for value in filters.values())
    candidate_count = max(top_k, len(index.chunks) if filtered else candidate_limit)
    keyword_results = index.search(query, top_k=candidate_count, **filters)
    semantic_options = {
        "top_k": candidate_count,
        "product_id": product_id,
        "document_id": document_id,
        "chunk_type": chunk_type,
        "model_name": model_name,
        "encoder": encoder,
    }
    if index_bundle is not None:
        semantic_options["index_bundle"] = index_bundle
    semantic_results = semantic_retriever(repository_root, query, **semantic_options)
    return reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )
