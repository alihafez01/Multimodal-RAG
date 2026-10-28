"""Build a persistent FAISS cosine-similarity index from chunks.jsonl."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from digital_product_rag.retrieval.embeddings import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from digital_product_rag.retrieval.vector_store import VectorStoreError, build_vector_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Sentence Transformers model ID (or set RAG_EMBEDDING_MODEL).")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    chosen_model = args.model or os.environ.get("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    print(f"Embedding model: {chosen_model}")
    print("Device: CPU")
    try:
        result = build_vector_index(
            REPOSITORY_ROOT, model_name=args.model, batch_size=args.batch_size
        )
    except (VectorStoreError, ValueError, OSError) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 2
    if result["model_changed"] or result["dimension_changed"]:
        print(
            "Previous index configuration changed: "
            f"model {result['previous_model']!r} -> {result['model_name']!r}, "
            f"dimension {result['previous_dimension']!r} -> {result['dimension']}."
        )
    print("FAISS index build summary")
    print(f"  Model: {result['model_name']}")
    print(f"  Embedding dimension: {result['dimension']}")
    print(f"  Similarity: normalized inner product (cosine similarity)")
    print(f"  Indexed vectors: {result['vector_count']}")
    print(f"  Source chunks: {result['source_chunk_count']}")
    print(f"  Invalid chunks skipped: {result['skipped_invalid_chunks']}")
    print(f"  Index: {result['index_path']}")
    print(f"  Vector mapping: {result['mapping_path']}")
    print(f"  Index metadata: {result['info_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
