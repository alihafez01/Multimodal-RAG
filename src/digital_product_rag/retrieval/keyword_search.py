"""In-memory BM25 keyword retrieval over the canonical chunk JSONL."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ..schemas import SearchResult
from .vector_store import VALID_CHUNK_TYPES, _valid_chunk


_TOKEN_RE = re.compile(r"#\d+(?:[a-z]+)?|[^\W_]+(?:[-/'\u2044][^\W_]+)*", re.UNICODE)
_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
})


class KeywordSearchError(RuntimeError):
    """Raised when the BM25 corpus cannot be safely loaded or searched."""


def tokenize(text: str) -> list[str]:
    """Normalize case and retain stitch terms, hook sizes, fractions, and hyphenated IDs."""
    if not isinstance(text, str) or not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [token for token in _TOKEN_RE.findall(normalized) if token not in _STOP_WORDS]


def has_query_terms(query: str) -> bool:
    return bool(tokenize(query))


class KeywordSearchIndex:
    """Load and validate the chunk corpus once, then reuse its BM25 index."""

    def __init__(self, chunks_path: Path):
        if not chunks_path.is_file():
            raise KeywordSearchError(f"Chunk dataset not found: {chunks_path}.")
        try:
            rows: list[dict[str, Any]] = []
            with chunks_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise KeywordSearchError(
                            f"Malformed chunk JSON on line {line_number} in {chunks_path}."
                        ) from exc
                    if not isinstance(row, dict):
                        raise KeywordSearchError(f"Chunk line {line_number} must contain an object.")
                    rows.append(row)
        except UnicodeDecodeError as exc:
            raise KeywordSearchError("Chunk dataset must be UTF-8 JSON Lines.") from exc
        except OSError as exc:
            raise KeywordSearchError(f"Could not read chunk dataset {chunks_path}: {exc}") from exc

        seen: set[str] = set()
        valid: list[dict[str, Any]] = []
        skipped = 0
        for line_number, row in enumerate(rows, start=1):
            chunk_id = row.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id in seen:
                raise KeywordSearchError(f"Duplicate chunk_id {chunk_id!r} on line {line_number}.")
            if isinstance(chunk_id, str):
                seen.add(chunk_id)
            required_result_fields = ("product_title",)
            if (
                not _valid_chunk(row)
                or any(not isinstance(row.get(field), str) or not row[field].strip() for field in required_result_fields)
            ):
                skipped += 1
                continue
            valid.append(row)
        if not valid:
            raise KeywordSearchError("No valid non-empty chunks are available for keyword search.")

        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:
            raise KeywordSearchError(
                "BM25 retrieval requires rank-bm25. Install it in the active global Python "
                "interpreter with: python -m pip install rank-bm25"
            ) from exc

        self.chunks = tuple(valid)
        self.skipped_invalid_chunks = skipped
        self._tokens = tuple(tokenize(row["text"]) for row in self.chunks)
        if not any(self._tokens):
            raise KeywordSearchError("No searchable tokens were found in the valid chunk dataset.")
        self.vocabulary = frozenset(token for tokens in self._tokens for token in tokens)
        self._bm25 = BM25Okapi([list(tokens) for tokens in self._tokens])
        self.product_ids = frozenset(row["product_id"] for row in self.chunks)
        self.document_ids = frozenset(row["document_id"] for row in self.chunks)

    def validate_filters(
        self,
        *,
        product_id: str | None = None,
        document_id: str | None = None,
        chunk_type: str | None = None,
    ) -> None:
        for name, value in (("product_id", product_id), ("document_id", document_id)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when supplied.")
        if product_id is not None and product_id not in self.product_ids:
            raise ValueError(f"Unknown product_id {product_id!r} in the chunk dataset.")
        if document_id is not None and document_id not in self.document_ids:
            raise ValueError(f"Unknown document_id {document_id!r} in the chunk dataset.")
        if chunk_type is not None and chunk_type not in VALID_CHUNK_TYPES:
            raise ValueError("chunk_type must be 'text', 'image_description', or None.")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        product_id: str | None = None,
        document_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.validate_filters(product_id=product_id, document_id=document_id, chunk_type=chunk_type)
        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        candidates: list[tuple[float, str, int]] = []
        query_terms = set(query_tokens)
        for index, (row, document_tokens, score) in enumerate(zip(self.chunks, self._tokens, scores)):
            if not query_terms.intersection(document_tokens):
                continue
            if product_id is not None and row["product_id"] != product_id:
                continue
            if document_id is not None and row["document_id"] != document_id:
                continue
            if chunk_type is not None and row["chunk_type"] != chunk_type:
                continue
            candidates.append((float(score), row["chunk_id"], index))
        candidates.sort(key=lambda item: (-item[0], item[1]))

        results: list[SearchResult] = []
        for rank, (score, _, index) in enumerate(candidates[:top_k], start=1):
            row = self.chunks[index]
            results.append(SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                product_id=row["product_id"],
                product_title=row["product_title"],
                product_category=row.get("product_category", "unknown"),
                product_version=row.get("product_version"),
                chunk_type=row["chunk_type"],
                section_title=row.get("section_title"),
                text=row["text"],
                filename=row["filename"],
                relative_path=row["relative_path"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                source_pages=tuple(row["source_pages"]),
                asset_id=row.get("asset_id"),
                asset_path=row.get("asset_path"),
                similarity_score=None,
                metadata_status=row.get("metadata_status"),
                bm25_score=score,
                bm25_rank=rank,
                retrieval_mode="keyword",
            ))
        return results


def search_keyword(
    repository_root: Path,
    query: str,
    *,
    top_k: int = 5,
    product_id: str | None = None,
    document_id: str | None = None,
    chunk_type: str | None = None,
) -> list[SearchResult]:
    """Convenience one-shot BM25 search; CLIs should reuse ``KeywordSearchIndex``."""
    index = KeywordSearchIndex(repository_root / "data" / "processed" / "chunks.jsonl")
    return index.search(
        query, top_k=top_k, product_id=product_id, document_id=document_id, chunk_type=chunk_type
    )
