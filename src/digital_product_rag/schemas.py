"""Small shared data structures for catalog, chunk, and retrieval records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    category: str
    version: str | None
    document_ids: tuple[str, ...]
    metadata_status: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    product_id: str
    product_title: str
    product_category: str
    product_version: str | None
    filename: str
    relative_path: str
    page_start: int
    page_end: int
    source_pages: tuple[int, ...]
    section_title: str | None
    chunk_type: str
    text: str
    asset_id: str | None = None
    asset_path: str | None = None
    metadata_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record, with source pages represented as an array."""
        result = asdict(self)
        result["source_pages"] = list(self.source_pages)
        return result


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    product_id: str
    product_title: str
    product_category: str
    product_version: str | None
    chunk_type: str
    section_title: str | None
    text: str
    filename: str
    relative_path: str
    page_start: int
    page_end: int
    source_pages: tuple[int, ...]
    asset_id: str | None
    asset_path: str | None
    similarity_score: float | None
    metadata_status: str | None = None
    semantic_score: float | None = None
    bm25_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    retrieval_mode: str = "semantic"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_pages"] = list(self.source_pages)
        return result
