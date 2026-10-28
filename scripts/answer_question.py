"""Retrieve cited evidence and optionally ask an already installed local Ollama model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
# Retrieval and evidence-only answering must use the existing local embedding cache only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from digital_product_rag.answering.answer_generator import (  # noqa: E402
    AnswerResponse,
    EvidenceCitation,
    SAFE_ABSTENTION_MESSAGE,
    answer_question as generate_answer,
    is_sufficient_evidence,
)
from digital_product_rag.application import (  # noqa: E402
    load_catalog as _load_catalog,
    product_title_focus as _product_title_focus,
    retrieve as _retrieve,
    validate_filters as _validate_filters,
)
from digital_product_rag.retrieval.keyword_search import KeywordSearchError  # noqa: E402
from digital_product_rag.retrieval.vector_store import VectorStoreError  # noqa: E402


def _nonempty_query(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("query must not be empty or whitespace-only")
    return value.strip()


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


def _page_label(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _print_evidence(citations: Sequence[EvidenceCitation]) -> None:
    if not citations:
        print("No useful results were retrieved for this query and filter combination.")
        return
    print(f"Retrieved {len(citations)} evidence passage(s):")
    for item in citations:
        print(
            f"\n[{item.reference}] {item.product_title} ({item.product_id})\n"
            f"  Source: {item.filename} | document_id={item.document_id} | "
            f"pages={_page_label(item.page_start, item.page_end)} | "
            f"section={item.section_title or '(no section title)'} | chunk_id={item.chunk_id}"
        )
        if item.chunk_type == "image_description":
            print(f"  Image asset: asset_id={item.asset_id} | asset_path={item.asset_path}")
        print("  Passage:")
        print(item.text)


def _print_citation_map(citations: Sequence[EvidenceCitation]) -> None:
    if not citations:
        return
    print("\nValidated citation sources (resolved from chunk metadata):")
    for item in citations:
        print(
            f"  [{item.reference}] {item.filename} | document_id={item.document_id} | "
            f"pages={_page_label(item.page_start, item.page_end)} | "
            f"section={item.section_title or '(no section title)'} | chunk_id={item.chunk_id} | "
            f"product_id={item.product_id} | product={item.product_title}"
        )
        if item.chunk_type == "image_description":
            print(f"      asset_id={item.asset_id} | asset_path={item.asset_path}")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, type=_nonempty_query)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--retrieval-mode", choices=("semantic", "keyword", "hybrid"), default="hybrid")
    parser.add_argument("--answer-mode", choices=("evidence", "ollama"), default="evidence")
    parser.add_argument("--product-id")
    parser.add_argument("--document-id")
    parser.add_argument("--chunk-type", choices=("text", "image_description"))
    parser.add_argument("--generation-model", help="Exact installed local model; defaults to RAG_GENERATION_MODEL.")
    parser.add_argument("--ollama-host", default=None, help="Defaults to OLLAMA_HOST or http://localhost:11434.")
    parser.add_argument("--timeout", type=_positive_float, default=120.0)
    args = parser.parse_args(argv)

    try:
        products, product_by_document = _load_catalog(REPOSITORY_ROOT)
        _validate_filters(
            products, product_by_document, product_id=args.product_id, document_id=args.document_id
        )
        focus_ids, anchors = _product_title_focus(args.query, products)
        explicit_ids: set[str] | None = None
        if args.product_id:
            explicit_ids = {args.product_id}
        elif args.document_id:
            explicit_ids = {product_by_document[args.document_id]}
        if focus_ids is not None and explicit_ids is not None and not focus_ids.intersection(explicit_ids):
            print(
                "No useful results: the query's catalog-title terms "
                f"({', '.join(anchors)}) do not match the selected product/document."
            )
            return 0

        if focus_ids is not None and not focus_ids:
            results: list[Any] = []
            print(
                "No catalog product title matches the product-specific query terms "
                f"({', '.join(anchors)}); unrelated product passages will not be used."
            )
        else:
            search_all = focus_ids is not None and explicit_ids is None
            results = _retrieve(
                REPOSITORY_ROOT,
                args.query,
                retrieval_mode=args.retrieval_mode,
                top_k=args.top_k,
                product_id=args.product_id,
                document_id=args.document_id,
                chunk_type=args.chunk_type,
                search_all_for_product_focus=search_all,
            )
            if focus_ids is not None:
                results = [result for result in results if result.product_id in focus_ids]
            results = results[:args.top_k]
            for result in results:
                product = products.get(result.product_id)
                if product is None or result.document_id not in product["document_ids"]:
                    raise ValueError(
                        f"Chunk {result.chunk_id!r} conflicts with the product catalog mapping; "
                        "rebuild chunks after correcting the catalog."
                    )

        if anchors and focus_ids:
            print(
                "Catalog title focus: " + ", ".join(anchors) + " -> "
                + ", ".join(f"{products[item]['title']} ({item})" for item in sorted(focus_ids))
            )
        response = generate_answer(
            args.query,
            results,
            mode=args.answer_mode,
            model=args.generation_model,
            ollama_host=args.ollama_host,
            timeout=args.timeout,
            citation_prefix="E",
        )
    except (KeywordSearchError, VectorStoreError, ValueError, OSError, RuntimeError) as exc:
        print(f"Answer request failed: {exc}", file=sys.stderr)
        return 2

    print(f"\nEvidence-only mode ({args.retrieval_mode} retrieval; no generated synthesis).") if args.answer_mode == "evidence" else None
    if not is_sufficient_evidence(args.query, results):
        print("Relevance check: passages did not meet the evidence sufficiency rule; no answer is claimed.")
    _print_evidence(response.citations or ())
    if args.answer_mode == "ollama":
        if response.status == "answered":
            print(f"\nAnswer (local Ollama model {response.model}):\n{response.answer}")
        elif response.abstained:
            print(f"\nAbstained: {response.abstention_reason or SAFE_ABSTENTION_MESSAGE}")
        else:
            print(
                f"\nLocal answer generation was unavailable ({response.status}).\n"
                "Retrieved passages are shown above; no generated answer is being reported as verified."
            )
    _print_citation_map(response.citations)
    if args.answer_mode == "ollama" and response.status in {"ollama_unavailable", "llm_failed"}:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
