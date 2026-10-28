"""Sentence Transformer model loading and normalized text embedding helpers."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from ..ingestion.progress import ProgressReporter


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Raised when a sentence embedding model cannot be loaded or used."""


def configured_model_name(model_name: str | None = None) -> str:
    name = model_name or os.environ.get("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if not isinstance(name, str) or not name.strip():
        raise EmbeddingError("Embedding model name must be a non-empty string.")
    return name.strip()


def load_sentence_transformer(model_name: str, *, device: str = "cpu") -> Any:
    """Load the requested model from its normal Hugging Face cache or remote ID."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # import can fail through a transitive torch/transformers issue
        raise EmbeddingError(
            "Sentence Transformers could not be imported. Check the installed "
            "sentence-transformers and PyTorch packages."
        ) from exc
    try:
        # Prefer a complete normal-cache copy so an offline run does not unnecessarily
        # probe the model host for optional adapter metadata.
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
        return model
    except Exception as cache_error:
        cache_error_type = type(cache_error).__name__
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=False)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model {model_name!r}. If it is not cached, "
            "check access to the Hugging Face model repository and retry. "
            f"Cached load detail: {cache_error_type}; download detail: {type(exc).__name__}."
        ) from exc
    return model


def encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode text in batches and return finite, L2-normalized float32 vectors."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise EmbeddingError("Embedding input must contain only non-empty text strings.")

    vectors: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    reporter = ProgressReporter("Embedding chunks", total_batches, enabled=show_progress)
    if show_progress:
        reporter.start(f"Encoding {len(texts)} chunks in batches of {batch_size}")
    try:
        for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = list(texts[start:start + batch_size])
            try:
                encoded = model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingError(f"Embedding batch {batch_number}/{total_batches} failed: {exc}") from exc
            array = np.asarray(encoded, dtype=np.float32)
            if array.ndim == 1 and len(batch) == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[0] != len(batch) or array.shape[1] < 1:
                raise EmbeddingError(
                    f"Embedding batch {batch_number} returned an invalid shape {array.shape}."
                )
            if not np.isfinite(array).all():
                raise EmbeddingError(f"Embedding batch {batch_number} contains non-finite values.")
            vectors.append(array)
            if show_progress:
                reporter.update(batch_number, f"batch {batch_number}/{total_batches}")
    finally:
        if show_progress:
            reporter.finish("Embedding chunks")

    combined = np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)
    if not np.isfinite(combined).all():
        raise EmbeddingError("Embeddings contain non-finite values.")
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise EmbeddingError("Embeddings contain zero-length or invalid vectors.")
    combined /= norms
    return np.ascontiguousarray(combined, dtype=np.float32)
