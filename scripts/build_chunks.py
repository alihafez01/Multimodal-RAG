"""Create the product catalog and deterministic searchable chunk dataset."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from digital_product_rag.chunking.chunker import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_OVERLAP_CHARS,
    ChunkingError,
    build_chunks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    args = parser.parse_args()
    try:
        result = build_chunks(
            REPOSITORY_ROOT,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            min_chars=args.min_chars,
        )
    except (ChunkingError, ValueError, OSError) as exc:
        print(f"Chunk build failed: {exc}", file=sys.stderr)
        return 2

    text_lengths = [len(row["text"]) for row in result.chunks if row["chunk_type"] == "text"]
    visual_count = sum(row["chunk_type"] == "image_description" for row in result.chunks)
    print("Chunk build summary")
    print(f"  Documents in inventory: {len(result.catalog.product_by_document)}")
    print(f"  Products in catalog: {len(result.catalog.products)}")
    print(f"  New products added: {result.catalog.generated_products}")
    print(f"  Product catalog: {result.catalog.path}")
    print(f"  Page source: {result.source_name}")
    if result.fallback_reason:
        print(f"  Enriched-page fallback reason: {result.fallback_reason}")
    print(f"  Pages examined: {result.pages_examined}")
    print(f"  Blank pages (diagnostic): {result.blank_pages}")
    print(f"  Page extraction errors: {result.page_errors}")
    print(f"  Non-valid inventory documents skipped: {result.skipped_documents}")
    print(f"  Text chunks: {len(text_lengths)}")
    print(f"  Image-description chunks: {visual_count}")
    print(f"  Short text fragments skipped: {result.skipped_short_text}")
    print(f"  Invalid or missing asset references: {result.invalid_assets}")
    print(f"  Visual records ignored (no eligible description): {result.ignored_visual_records}")
    if text_lengths:
        print(
            "  Text chunk chars (min/mean/max): "
            f"{min(text_lengths)}/{statistics.mean(text_lengths):.1f}/{max(text_lengths)}"
        )
    all_lengths = [len(row["text"]) for row in result.chunks]
    if all_lengths:
        print(
            "  All chunk chars (min/mean/max): "
            f"{min(all_lengths)}/{statistics.mean(all_lengths):.1f}/{max(all_lengths)}"
        )
    print(f"  Final chunks: {len(result.chunks)}")
    print(f"  Output: {REPOSITORY_ROOT / 'data' / 'processed' / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
