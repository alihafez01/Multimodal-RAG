"""Shared, read-only application helpers for CLI, UI, and evaluation entry points."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .answering.answer_generator import AnswerResponse, answer_question
from .ingestion.common import ManifestError, read_manifest
from .retrieval.hybrid_search import DEFAULT_CANDIDATE_LIMIT, search_hybrid
from .retrieval.keyword_search import KeywordSearchError, KeywordSearchIndex, tokenize
from .retrieval.query_planning import QueryPlan, combine_query_results, plan_query, prioritize_sequence
from .retrieval.vector_store import VectorStoreError, search as semantic_search


GENERIC_TITLE_TERMS = frozenset({
    "beginner", "crochet", "digital", "download", "diy", "instructions", "pattern", "pdf",
    "plan", "product", "template", "tutorial",
})


def load_catalog(repository_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load the current product catalog and ensure every inventory document maps once."""
    path = repository_root / "config" / "product_catalog.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Product catalog not found: {path}.") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read product catalog {path}: {exc}.") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Product catalog must be an object with schema_version 1.")
    raw_products = data.get("products")
    if not isinstance(raw_products, list) or not raw_products:
        raise ValueError("Product catalog 'products' must be a non-empty list.")

    try:
        documents = read_manifest(
            repository_root / "data" / "processed" / "manifest.jsonl",
            repository_root / "Dataset",
        )
    except ManifestError as exc:
        raise ValueError(f"Could not validate catalog document mappings: {exc}") from exc
    known_document_ids = {document.document_id for document in documents}
    products: dict[str, dict[str, Any]] = {}
    product_by_document: dict[str, str] = {}
    for index, row in enumerate(raw_products, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Product catalog entry {index} must be an object.")
        product_id, title, document_ids = row.get("product_id"), row.get("title"), row.get("document_ids")
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError(f"Product catalog entry {index} has no valid product_id.")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Product catalog entry {index} has no valid title.")
        category = row.get("category", "unknown")
        version = row.get("version")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Product {product_id!r} has an invalid category.")
        if version is not None and not isinstance(version, str):
            raise ValueError(f"Product {product_id!r} version must be a string or null.")
        if product_id in products:
            raise ValueError(f"Product catalog contains duplicate product_id {product_id!r}.")
        if (
            not isinstance(document_ids, list)
            or not document_ids
            or any(not isinstance(item, str) or not item.strip() for item in document_ids)
            or len(set(document_ids)) != len(document_ids)
        ):
            raise ValueError(f"Product {product_id!r} must map to a non-empty list of unique document IDs.")
        products[product_id] = row
        for document_id in document_ids:
            if document_id not in known_document_ids:
                raise ValueError(f"Product {product_id!r} maps to unknown document_id {document_id!r}.")
            if document_id in product_by_document:
                raise ValueError(f"Document {document_id!r} is mapped to multiple catalog products.")
            product_by_document[document_id] = product_id
    unmapped = sorted(known_document_ids - set(product_by_document))
    if unmapped:
        raise ValueError("Product catalog has unmapped source documents: " + ", ".join(unmapped))
    return products, product_by_document


def load_document_inventory(repository_root: Path):
    """Return validated inventory records without scanning or modifying the source PDFs."""
    try:
        return read_manifest(
            repository_root / "data" / "processed" / "manifest.jsonl",
            repository_root / "Dataset",
        )
    except ManifestError as exc:
        raise ValueError(f"Could not read document inventory: {exc}") from exc


def validate_filters(
    products: dict[str, dict[str, Any]],
    product_by_document: dict[str, str],
    *,
    product_id: str | None,
    document_id: str | None,
) -> None:
    """Reject unknown or contradictory product/document selection IDs."""
    if product_id is not None and product_id not in products:
        raise ValueError(f"Unknown product_id {product_id!r} in config/product_catalog.json.")
    if document_id is not None and document_id not in product_by_document:
        raise ValueError(f"Unknown document_id {document_id!r} in config/product_catalog.json.")
    if product_id is not None and document_id is not None and product_by_document[document_id] != product_id:
        raise ValueError(
            f"Filter mismatch: document_id {document_id!r} is not mapped to product_id {product_id!r}."
        )


def product_title_focus(
    query: str,
    products: dict[str, dict[str, Any]],
) -> tuple[frozenset[str] | None, tuple[str, ...]]:
    """Find exact, relatively rare catalog-title anchors such as 'sweater'."""
    query_terms = set(tokenize(query))
    title_terms = {product_id: set(tokenize(row["title"])) for product_id, row in products.items()}
    frequency = Counter(term for terms in title_terms.values() for term in terms)
    maximum_frequency = max(1, len(products) // 4)
    anchors = tuple(sorted(
        term for term in query_terms
        if term not in GENERIC_TITLE_TERMS and term in frequency and frequency[term] <= maximum_frequency
    ))
    if not anchors:
        return None, ()
    anchor_set = set(anchors)
    matching = frozenset(
        product_id for product_id, terms in title_terms.items() if anchor_set.issubset(terms)
    )
    return matching, anchors


def _chunk_line_count(path: Path) -> int:
    if not path.is_file():
        raise VectorStoreError(f"Chunk dataset not found: {path}.")
    try:
        with path.open("r", encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip())
    except (OSError, UnicodeDecodeError) as exc:
        raise VectorStoreError(f"Could not inspect chunk dataset {path}: {exc}.") from exc
    if count < 1:
        raise VectorStoreError(f"Chunk dataset is empty: {path}.")
    return count


def retrieve(
    repository_root: Path,
    query: str,
    *,
    retrieval_mode: str,
    top_k: int,
    product_id: str | None,
    document_id: str | None,
    chunk_type: str | None,
    search_all_for_product_focus: bool = False,
    keyword_index: KeywordSearchIndex | None = None,
    index_bundle: tuple[Any, dict[int, dict[str, Any]], dict[str, Any]] | None = None,
    encoder: Any | None = None,
) -> list[Any]:
    """Use the existing semantic, BM25, or hybrid retriever with optional cached resources."""
    if retrieval_mode not in {"semantic", "keyword", "hybrid"}:
        raise ValueError("retrieval_mode must be semantic, keyword, or hybrid.")
    filters = {"product_id": product_id, "document_id": document_id, "chunk_type": chunk_type}
    chunk_path = repository_root / "data" / "processed" / "chunks.jsonl"
    if retrieval_mode in {"keyword", "hybrid"}:
        keyword_index = keyword_index or KeywordSearchIndex(chunk_path)
        keyword_index.validate_filters(**filters)
    retrieval_k = top_k
    if search_all_for_product_focus:
        retrieval_k = len(keyword_index.chunks) if keyword_index else _chunk_line_count(chunk_path)
    if retrieval_mode == "semantic":
        return semantic_search(
            repository_root, query, top_k=retrieval_k, encoder=encoder, index_bundle=index_bundle, **filters
        )
    if retrieval_mode == "keyword":
        assert keyword_index is not None
        return keyword_index.search(query, top_k=retrieval_k, **filters)
    assert keyword_index is not None
    return search_hybrid(
        repository_root,
        query,
        top_k=retrieval_k,
        candidate_limit=max(DEFAULT_CANDIDATE_LIMIT, retrieval_k),
        keyword_index=keyword_index,
        encoder=encoder,
        index_bundle=index_bundle,
        **filters,
    )


def retrieve_for_question(
    repository_root: Path,
    query: str,
    *,
    products: dict[str, dict[str, Any]],
    product_by_document: dict[str, str],
    retrieval_mode: str,
    top_k: int,
    product_id: str | None = None,
    document_id: str | None = None,
    chunk_type: str | None = None,
    evidence_selection: str = "auto",
    keyword_index: KeywordSearchIndex | None = None,
    index_bundle: tuple[Any, dict[int, dict[str, Any]], dict[str, Any]] | None = None,
    encoder: Any | None = None,
) -> tuple[list[Any], QueryPlan, tuple[str, ...], frozenset[str] | None]:
    """Plan bounded subqueries and reuse the retrievers without broadening filters."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if evidence_selection not in {"auto", "all", "text", "image_description"}:
        raise ValueError("evidence_selection must be auto, all, text, or image_description.")
    if evidence_selection in {"text", "image_description"}:
        if chunk_type is not None and chunk_type != evidence_selection:
            raise ValueError("Conflicting evidence-type filters.")
        chunk_type = evidence_selection
    validate_filters(products, product_by_document, product_id=product_id, document_id=document_id)
    keyword_index = keyword_index or KeywordSearchIndex(repository_root / "data/processed/chunks.jsonl")
    keyword_index.validate_filters(product_id=product_id, document_id=document_id, chunk_type=chunk_type)
    plan = plan_query(query, vocabulary=keyword_index.vocabulary, product_titles=(p["title"] for p in products.values()))
    focus_ids, anchors = product_title_focus(plan.normalized_query, products)
    explicit_ids = {product_id} if product_id else (
        {product_by_document[document_id]} if document_id else None
    )
    if focus_ids is not None and explicit_ids is not None and not focus_ids.intersection(explicit_ids):
        results: list[Any] = []
    elif focus_ids is not None and not focus_ids:
        results = []
    else:
        effective_product = product_id
        if effective_product is None and focus_ids and len(focus_ids) == 1:
            effective_product = next(iter(focus_ids))
        candidate_k = max(15, top_k * 3)
        rankings: list[list[Any]] = []
        # The substantive procedure comes first; identification still gets its own slot.
        for part in sorted(plan.parts, key=lambda item: item.intent != "sequence"):
            selected_type = chunk_type or (None if part.intent == "visual" else "text")
            candidates = retrieve(
                repository_root, part.query, retrieval_mode=retrieval_mode,
                top_k=candidate_k, product_id=effective_product, document_id=document_id,
                chunk_type=selected_type,
                search_all_for_product_focus=focus_ids is not None and effective_product is None,
                keyword_index=keyword_index, index_bundle=index_bundle, encoder=encoder,
            )
            if focus_ids is not None:
                candidates = [item for item in candidates if item.product_id in focus_ids]
            candidates = prioritize_sequence(candidates, part)
            if part.intent == "product":
                candidates.sort(key=lambda item: not (
                    " ".join(item.product_title.casefold().split()) in " ".join(item.text.casefold().split())
                ))
            # Explicit all admits visual supplements when the written pool is short.
            # A visual subquestion always searches both types; explicit filters win.
            if evidence_selection == "all" and chunk_type is None and selected_type == "text" and len(candidates) < top_k:
                supplemental = retrieve(
                    repository_root, part.query, retrieval_mode=retrieval_mode,
                    top_k=candidate_k, product_id=effective_product, document_id=document_id,
                    chunk_type="image_description", keyword_index=keyword_index,
                    index_bundle=index_bundle, encoder=encoder,
                    search_all_for_product_focus=focus_ids is not None and effective_product is None,
                )
                candidates.extend(item for item in supplemental if focus_ids is None or item.product_id in focus_ids)
            rankings.append(candidates)
        results = combine_query_results(rankings, top_k=top_k)
        for result in results:
            product = products.get(result.product_id)
            if product is None or result.document_id not in product["document_ids"]:
                raise ValueError(
                    f"Chunk {result.chunk_id!r} conflicts with the product catalog mapping."
                )
    return results, plan, anchors, focus_ids


def query_knowledge_base(
    repository_root: Path,
    query: str,
    *,
    products: dict[str, dict[str, Any]],
    product_by_document: dict[str, str],
    retrieval_mode: str,
    answer_mode: str,
    top_k: int,
    product_id: str | None = None,
    document_id: str | None = None,
    chunk_type: str | None = None,
    evidence_selection: str = "auto",
    keyword_index: KeywordSearchIndex | None = None,
    index_bundle: tuple[Any, dict[int, dict[str, Any]], dict[str, Any]] | None = None,
    encoder: Any | None = None,
    model: str | None = None,
    ollama_host: str | None = None,
    timeout: float = 120.0,
) -> tuple[list[Any], AnswerResponse, tuple[str, ...], frozenset[str] | None]:
    """Shared UI/CLI entry point using the same plan and citation-checked generator."""
    results, plan, anchors, focus_ids = retrieve_for_question(
        repository_root, query, products=products, product_by_document=product_by_document,
        retrieval_mode=retrieval_mode, top_k=top_k, product_id=product_id,
        document_id=document_id, chunk_type=chunk_type, evidence_selection=evidence_selection,
        keyword_index=keyword_index, index_bundle=index_bundle, encoder=encoder,
    )
    response = answer_question(
        plan.normalized_query, results, mode=answer_mode, citation_prefix="E",
        model=model, ollama_host=ollama_host, timeout=timeout,
    )
    return results, response, anchors, focus_ids
