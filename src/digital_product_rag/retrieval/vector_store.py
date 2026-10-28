"""Persistent FAISS index building, consistency checks, and semantic search."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..ingestion.common import atomic_write_bytes
from ..schemas import SearchResult
from .embeddings import (
    EmbeddingError,
    configured_model_name,
    encode_texts,
    load_sentence_transformer,
)


INDEX_SCHEMA_VERSION = 1
SIMILARITY_STRATEGY = "normalized_inner_product_cosine"
VALID_CHUNK_TYPES = {"text", "image_description"}


class VectorStoreError(RuntimeError):
    """Raised for invalid, stale, or inconsistent vector-index operations."""


def index_paths(repository_root: Path) -> dict[str, Path]:
    directory = repository_root / "data" / "indexes"
    return {
        "directory": directory,
        "index": directory / "chunks.faiss",
        "mapping": directory / "vector_metadata.jsonl",
        "info": directory / "index_info.json",
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_model_cache_locations(repository_root: Path) -> None:
    """Prevent configured model caches from placing weights anywhere in the repository."""
    names = (
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME",
    )
    root = repository_root.resolve()
    for name in names:
        configured = os.environ.get(name)
        if not configured:
            continue
        cache_path = Path(configured).expanduser().resolve()
        if cache_path == root or cache_path.is_relative_to(root):
            raise VectorStoreError(
                f"{name} points inside the repository ({cache_path}); use the standard external model cache."
            )


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VectorStoreError(f"Malformed {label} JSON on line {line_number}.") from exc
                if not isinstance(record, dict):
                    raise VectorStoreError(f"{label} line {line_number} must contain an object.")
                records.append(record)
    except OSError as exc:
        raise VectorStoreError(f"Could not read {label} {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise VectorStoreError(f"{label} must be UTF-8 JSON Lines.") from exc
    return records


def _valid_chunk(row: dict[str, Any]) -> bool:
    required_strings = ("chunk_id", "document_id", "product_id", "text", "filename", "relative_path")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required_strings):
        return False
    if row.get("chunk_type") not in VALID_CHUNK_TYPES:
        return False
    pages = row.get("source_pages")
    if (
        not isinstance(pages, list) or not pages or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in pages
        )
    ):
        return False
    if pages != sorted(set(pages)):
        return False
    start, end = row.get("page_start"), row.get("page_end")
    if (
        isinstance(start, bool) or not isinstance(start, int)
        or isinstance(end, bool) or not isinstance(end, int)
        or start != min(pages) or end != max(pages)
    ):
        return False
    if row["chunk_type"] == "image_description":
        return (
            isinstance(row.get("asset_id"), str) and bool(row["asset_id"])
            and isinstance(row.get("asset_path"), str) and bool(row["asset_path"])
            and start == end
        )
    return row.get("asset_id") is None and row.get("asset_path") is None


def _load_faiss() -> Any:
    try:
        import faiss
    except Exception as exc:
        raise VectorStoreError(
            "FAISS is unavailable. Install a compatible faiss-cpu package for this Python platform."
        ) from exc
    return faiss


def _load_embedding_model(model_name: str, encoder: Any | None) -> Any:
    if encoder is not None:
        return encoder
    try:
        return load_sentence_transformer(model_name, device="cpu")
    except EmbeddingError as exc:
        raise VectorStoreError(str(exc)) from exc


def _install_outputs(staged: dict[str, Path], targets: dict[str, Path], directory: Path) -> None:
    """Replace all index files after successful staging, rolling back on replacement errors."""
    backups_dir = Path(tempfile.mkdtemp(prefix=".index-backup-", dir=directory))
    backups: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for key, target in targets.items():
            if target.exists():
                backup = backups_dir / target.name
                shutil.copy2(target, backup)
                backups[key] = backup
        for key in ("index", "mapping", "info"):
            os.replace(staged[key], targets[key])
            replaced.append(key)
    except Exception:
        for key in reversed(replaced):
            backup = backups.get(key)
            if backup is not None and backup.exists():
                os.replace(backup, targets[key])
            else:
                targets[key].unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(backups_dir, ignore_errors=True)


def build_vector_index(
    repository_root: Path,
    *,
    model_name: str | None = None,
    batch_size: int = 32,
    encoder: Any | None = None,
    faiss_module: Any | None = None,
) -> dict[str, Any]:
    """Rebuild a cosine-similarity FAISS index for the current chunk file."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    repository_root = repository_root.resolve()
    chunks_path = repository_root / "data" / "processed" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise VectorStoreError(f"Chunk dataset not found: {chunks_path}. Run scripts/build_chunks.py first.")
    chunk_bytes = chunks_path.read_bytes()
    source_fingerprint = _sha256_bytes(chunk_bytes)
    raw_chunks = _read_jsonl(chunks_path, "chunk dataset")
    valid_chunks: list[dict[str, Any]] = []
    skipped = 0
    seen: set[str] = set()
    for line_number, row in enumerate(raw_chunks, start=1):
        chunk_id = row.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id in seen:
            raise VectorStoreError(f"Duplicate chunk_id {chunk_id!r} on line {line_number}.")
        if isinstance(chunk_id, str):
            seen.add(chunk_id)
        if not _valid_chunk(row):
            skipped += 1
            continue
        valid_chunks.append(row)
    if not valid_chunks:
        raise VectorStoreError("No valid non-empty chunks are available; refusing to build an empty index.")
    # Sorting by the stable chunk ID makes vector assignment independent of JSONL order.
    valid_chunks.sort(key=lambda row: row["chunk_id"])
    chosen_model = configured_model_name(model_name)
    _validate_model_cache_locations(repository_root)
    model = _load_embedding_model(chosen_model, encoder)
    try:
        vectors = encode_texts(model, [row["text"] for row in valid_chunks], batch_size=batch_size)
    except EmbeddingError as exc:
        raise VectorStoreError(str(exc)) from exc
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if not callable(dimension_getter):
        dimension_getter = getattr(model, "get_sentence_embedding_dimension", None)
    expected_dimension = dimension_getter() if callable(dimension_getter) else None
    if expected_dimension is not None and int(expected_dimension) != vectors.shape[1]:
        raise VectorStoreError(
            f"Model reports embedding dimension {expected_dimension}, but encoded vectors have {vectors.shape[1]}."
        )
    faiss = faiss_module or _load_faiss()
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    try:
        index.add(vectors)
    except Exception as exc:
        raise VectorStoreError(f"FAISS rejected the embedding vectors: {exc}") from exc
    if index.ntotal != len(valid_chunks) or index.d != vectors.shape[1]:
        raise VectorStoreError("FAISS index size or dimension does not match the generated embeddings.")

    paths = index_paths(repository_root)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    previous_model = None
    previous_dimension = None
    if paths["info"].is_file():
        try:
            previous = json.loads(paths["info"].read_text(encoding="utf-8"))
            previous_model = previous.get("model_name")
            previous_dimension = previous.get("embedding_dimension")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass

    temporary_dir = Path(tempfile.mkdtemp(prefix=".index-build-", dir=paths["directory"]))
    try:
        staged = {
            "index": temporary_dir / paths["index"].name,
            "mapping": temporary_dir / paths["mapping"].name,
            "info": temporary_dir / paths["info"].name,
        }
        try:
            faiss.write_index(index, str(staged["index"]))
            check_index = faiss.read_index(str(staged["index"]))
        except Exception as exc:
            raise VectorStoreError(f"Could not serialize and verify the FAISS index: {exc}") from exc
        if check_index.ntotal != len(valid_chunks) or check_index.d != vectors.shape[1]:
            raise VectorStoreError("Serialized FAISS index failed its count/dimension verification.")

        provisional_build_id = hashlib.sha256(
            f"{source_fingerprint}:{chosen_model}:{vectors.shape[1]}:{len(valid_chunks)}".encode("utf-8")
        ).hexdigest()
        index_hash = _sha256_file(staged["index"])
        build_id = hashlib.sha256(f"{provisional_build_id}:{index_hash}".encode("utf-8")).hexdigest()
        mapping_bytes = "".join(
            json.dumps({"vector_id": vector_id, "chunk_id": row["chunk_id"], "build_id": build_id}, ensure_ascii=False) + "\n"
            for vector_id, row in enumerate(valid_chunks)
        ).encode("utf-8")
        staged["mapping"].write_bytes(mapping_bytes)
        mapping_hash = _sha256_bytes(mapping_bytes)
        info = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "model_name": chosen_model,
            "embedding_dimension": int(vectors.shape[1]),
            "similarity_strategy": SIMILARITY_STRATEGY,
            "vector_count": len(valid_chunks),
            "source_chunk_count": len(raw_chunks),
            "skipped_invalid_chunks": skipped,
            "chunks_sha256": source_fingerprint,
            "mapping_sha256": mapping_hash,
            "index_sha256": index_hash,
            "build_id": build_id,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        info_bytes = (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        staged["info"].write_bytes(info_bytes)
        file_targets = {key: paths[key] for key in ("index", "mapping", "info")}
        _install_outputs(staged, file_targets, paths["directory"])
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return {
        "index_path": paths["index"],
        "mapping_path": paths["mapping"],
        "info_path": paths["info"],
        "model_name": chosen_model,
        "dimension": int(vectors.shape[1]),
        "vector_count": len(valid_chunks),
        "source_chunk_count": len(raw_chunks),
        "skipped_invalid_chunks": skipped,
        "previous_model": previous_model,
        "previous_dimension": previous_dimension,
        "model_changed": previous_model is not None and previous_model != chosen_model,
        "dimension_changed": previous_dimension is not None and previous_dimension != vectors.shape[1],
        "build_id": build_id,
    }


def _load_index(repository_root: Path, *, model_name: str | None = None, faiss_module: Any | None = None):
    paths = index_paths(repository_root)
    missing = [str(paths[key]) for key in ("index", "mapping", "info") if not paths[key].is_file()]
    if missing:
        raise VectorStoreError(
            "FAISS index is incomplete or missing: " + ", ".join(missing)
            + ". Run scripts/build_vector_index.py first."
        )
    try:
        info_bytes = paths["info"].read_bytes()
        info = json.loads(info_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VectorStoreError(f"Index metadata is unreadable: {paths['info']}") from exc
    if not isinstance(info, dict) or info.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise VectorStoreError("Unsupported or malformed FAISS index metadata schema.")
    saved_model = info.get("model_name")
    if not isinstance(saved_model, str) or not saved_model:
        raise VectorStoreError("Index metadata does not identify its embedding model.")
    requested = model_name or os.environ.get("RAG_EMBEDDING_MODEL")
    if requested and requested != saved_model:
        raise VectorStoreError(
            f"Configured embedding model {requested!r} does not match indexed model {saved_model!r}. "
            "Set RAG_EMBEDDING_MODEL to the indexed model or rebuild the index."
        )
    chunks_path = repository_root / "data" / "processed" / "chunks.jsonl"
    if not chunks_path.is_file() or _sha256_file(chunks_path) != info.get("chunks_sha256"):
        raise VectorStoreError("The chunk dataset changed after indexing; rebuild data/indexes before searching.")
    if _sha256_file(paths["index"]) != info.get("index_sha256"):
        raise VectorStoreError("FAISS index checksum does not match index_info.json; rebuild the index.")
    if _sha256_file(paths["mapping"]) != info.get("mapping_sha256"):
        raise VectorStoreError("Vector mapping checksum does not match index_info.json; rebuild the index.")
    chunks = _read_jsonl(chunks_path, "chunk dataset")
    chunks_by_id = {row.get("chunk_id"): row for row in chunks if isinstance(row.get("chunk_id"), str)}
    if len(chunks_by_id) != len(chunks):
        raise VectorStoreError("Chunk dataset has missing or duplicate chunk IDs.")
    mapping_rows = _read_jsonl(paths["mapping"], "vector metadata")
    vector_to_chunk: dict[int, dict[str, Any]] = {}
    chunk_ids: set[str] = set()
    build_id = info.get("build_id")
    for line_number, row in enumerate(mapping_rows, start=1):
        vector_id, chunk_id = row.get("vector_id"), row.get("chunk_id")
        if (
            isinstance(vector_id, bool) or not isinstance(vector_id, int) or vector_id < 0
            or vector_id in vector_to_chunk
            or not isinstance(chunk_id, str) or chunk_id not in chunks_by_id
            or chunk_id in chunk_ids or row.get("build_id") != build_id
        ):
            raise VectorStoreError(f"Vector metadata line {line_number} is invalid or belongs to another index build.")
        vector_to_chunk[vector_id] = chunks_by_id[chunk_id]
        chunk_ids.add(chunk_id)
    expected_count = info.get("vector_count")
    if (
        isinstance(expected_count, bool) or not isinstance(expected_count, int)
        or expected_count < 1 or len(vector_to_chunk) != expected_count
        or set(vector_to_chunk) != set(range(expected_count))
    ):
        raise VectorStoreError("Vector metadata count or IDs do not match index_info.json.")
    faiss = faiss_module or _load_faiss()
    try:
        index = faiss.read_index(str(paths["index"]))
    except Exception as exc:
        raise VectorStoreError(f"Could not read FAISS index {paths['index']}: {exc}") from exc
    if index.ntotal != expected_count or index.d != info.get("embedding_dimension"):
        raise VectorStoreError("FAISS index count/dimension does not match its metadata mapping.")
    return index, vector_to_chunk, info


def load_index_bundle(repository_root: Path, *, model_name: str | None = None):
    """Load and fully validate a read-only FAISS bundle for a cached application session."""
    return _load_index(repository_root.resolve(), model_name=model_name)


def search(
    repository_root: Path,
    query: str,
    *,
    top_k: int = 5,
    product_id: str | None = None,
    document_id: str | None = None,
    chunk_type: str | None = None,
    model_name: str | None = None,
    encoder: Any | None = None,
    index_bundle: tuple[Any, dict[int, dict[str, Any]], dict[str, Any]] | None = None,
    faiss_module: Any | None = None,
) -> list[SearchResult]:
    """Search normalized text and image-description embeddings by cosine similarity."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    for name, value in (("product_id", product_id), ("document_id", document_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string when supplied.")
    if chunk_type is not None and chunk_type not in VALID_CHUNK_TYPES:
        raise ValueError("chunk_type must be 'text', 'image_description', or None.")
    repository_root = repository_root.resolve()
    if index_bundle is None:
        index, vector_to_chunk, info = _load_index(
            repository_root, model_name=model_name, faiss_module=faiss_module
        )
    else:
        index, vector_to_chunk, info = index_bundle
        requested_model = model_name or os.environ.get("RAG_EMBEDDING_MODEL")
        if requested_model and requested_model != info.get("model_name"):
            raise VectorStoreError(
                f"Configured embedding model {requested_model!r} does not match indexed model "
                f"{info.get('model_name')!r}. Set RAG_EMBEDDING_MODEL to the indexed model or rebuild the index."
            )
    indexed_chunks = list(vector_to_chunk.values())
    if product_id is not None and not any(row["product_id"] == product_id for row in indexed_chunks):
        raise ValueError(f"Unknown product_id {product_id!r} in the indexed chunk dataset.")
    if document_id is not None and not any(row["document_id"] == document_id for row in indexed_chunks):
        raise ValueError(f"Unknown document_id {document_id!r} in the indexed chunk dataset.")
    _validate_model_cache_locations(repository_root)
    model = _load_embedding_model(info["model_name"], encoder)
    query_vector = encode_texts(model, [query.strip()], batch_size=1, show_progress=False)
    if query_vector.shape[1] != index.d:
        raise VectorStoreError(
            f"Query embedding dimension {query_vector.shape[1]} does not match index dimension {index.d}."
        )
    filters = product_id is not None or document_id is not None or chunk_type is not None
    candidate_count = index.ntotal if filters else min(top_k, index.ntotal)
    try:
        scores, vector_ids = index.search(query_vector, candidate_count)
    except Exception as exc:
        raise VectorStoreError(f"FAISS search failed: {exc}") from exc
    results: list[SearchResult] = []
    for score, vector_id in zip(scores[0], vector_ids[0]):
        vector_id = int(vector_id)
        if vector_id < 0:
            continue
        chunk = vector_to_chunk.get(vector_id)
        if chunk is None:
            raise VectorStoreError(f"FAISS returned unmapped vector ID {vector_id}.")
        if product_id is not None and chunk.get("product_id") != product_id:
            continue
        if document_id is not None and chunk.get("document_id") != document_id:
            continue
        if chunk_type is not None and chunk.get("chunk_type") != chunk_type:
            continue
        results.append(SearchResult(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            product_id=chunk["product_id"],
            product_title=chunk["product_title"],
            product_category=chunk.get("product_category", "unknown"),
            product_version=chunk.get("product_version"),
            chunk_type=chunk["chunk_type"],
            section_title=chunk.get("section_title"),
            text=chunk["text"],
            filename=chunk["filename"],
            relative_path=chunk["relative_path"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            source_pages=tuple(chunk["source_pages"]),
            asset_id=chunk.get("asset_id"),
            asset_path=chunk.get("asset_path"),
            similarity_score=float(score),
            metadata_status=chunk.get("metadata_status"),
            semantic_score=float(score),
            semantic_rank=len(results) + 1,
            retrieval_mode="semantic",
        ))
        if len(results) >= top_k:
            break
    return results
